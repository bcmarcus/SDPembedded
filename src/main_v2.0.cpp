#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <esp_wpa2.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include <Wire.h>
#include "MPU6050.h"
#include "esp_task_wdt.h"
#include <HardwareSerial.h>

// ====== NETWORK CONFIGURATION ======
const bool useUCRWiFi = false; // true: use WPA2-Enterprise; false: use standard Wi-Fi

// Standard Wi-Fi credentials
const char* standardSSID     = "Imhatin43";
const char* standardPassword = "ILikeToType73";

// WPA2 Enterprise credentials
const char* secureSSID     = "UCR-SECURE";
const char* eapIdentity    = "bmarc018";
const char* eapPassword    = "City6464!";

// ====== SENSOR PIN ASSIGNMENTS ======
const int sigPinLeft    = 18;
const int sigPinBack    = 17;
const int sigPinRight   = 5;
const int sigPinForward = 16;

// ====== MOTOR UART CONFIGURATION ======
HardwareSerial motorSerial(2);  // Using UART2 on ESP32
const int motorRxPin = 19;  // RX pin connected to receiver's TX
const int motorTxPin = 23;  // TX pin connected to receiver's RX
SemaphoreHandle_t motorMutex = NULL;  // Mutex to protect motor UART access

// ====== SERVER CONFIGURATION ======
const char* serverIP   = "192.168.0.129";
const int   serverPort = 8642;

// ====== QUEUE SETTINGS ======
const int MSG_BUFFER_SIZE = 256;
const int SENSOR_QUEUE_SIZE = 32;  // Larger queue to prevent overflow

// ====== MPU6050 CONFIGURATION ======
MPU6050 mpu(Wire);
float gyroXoffset = 0.8279;
float gyroYoffset = 1.3698;
float gyroZoffset = -0.0624;
float accXoffset = -0.0548;
float accYoffset = -0.0352;
float accZoffset = 0.0515;

// ====== GLOBAL VARIABLES ======
QueueHandle_t sensorQueue = NULL;
SemaphoreHandle_t wifiMutex = NULL;
SemaphoreHandle_t queueMutex = NULL;  // New mutex to protect queue operations
const int WDT_TIMEOUT = 15;  // Increased watchdog timeout (seconds)

// ====== STATUS LED PINS ======
const int wifiLedPin = 2;    // Built-in LED for WiFi status
const int serverLedPin = 4;  // External LED for server connection status

// ====== LED BLINKING INTERVALS ======
const int WIFI_BLINK_INTERVAL = 600;      // Slow blinking for WiFi (500ms)
const int SERVER_BLINK_INTERVAL = 200;    // Fast blinking for server (250ms - twice as fast)

// ====== MOTOR COMMAND FUNCTIONS ======
void sendMotorCommand(const char* command) {
  if (xSemaphoreTake(motorMutex, 100 / portTICK_PERIOD_MS) == pdTRUE) {
    motorSerial.print(command);
    motorSerial.print("\n");  // Send newline to terminate command
    Serial.println("Sent motor command: " + String(command));
    
    // Check for response (optional)
    unsigned long startTime = millis();
    while (!motorSerial.available() && millis() - startTime < 500) {
      vTaskDelay(1);  // Small yield to prevent watchdog issues
    }
    
    if (motorSerial.available()) {
      String response = motorSerial.readStringUntil('\n');
      response.trim();
      Serial.println("Motor response: " + response);
    }
    
    xSemaphoreGive(motorMutex);
  } else {
    Serial.println("Warning: Could not acquire motor mutex");
  }
}

// ====== ULTRASONIC SENSOR FUNCTION ======
long measureDistance(int sigPin) {
  // Simple blocking ultrasonic measurement
  pinMode(sigPin, OUTPUT);
  digitalWrite(sigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(sigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(sigPin, LOW);
  
  pinMode(sigPin, INPUT);
  long duration = pulseIn(sigPin, HIGH, 30000); // 30ms timeout
  
  if (duration == 0) return 501; // Invalid reading (timeout)
  
  long distance = duration * 0.034 / 2; // Convert to cm
  
  // Filter unreasonable values
  if (distance > 500 || distance <= 0) {
    return 501;
  }
  
  return distance;
}

// ====== TASK 1: NETWORK TASK ======
void networkTask(void *pvParameters) {
  // Register with watchdog
  esp_task_wdt_add(NULL);
  
  WiFiClientSecure client;
  client.setInsecure();  // For self-signed certificates
  
  // Setup LEDs
  pinMode(wifiLedPin, OUTPUT);
  pinMode(serverLedPin, OUTPUT);
  digitalWrite(wifiLedPin, LOW);
  digitalWrite(serverLedPin, LOW);
  
  // Connect to WiFi
  Serial.println("NetworkTask: Connecting to WiFi...");
  
  if (xSemaphoreTake(wifiMutex, portMAX_DELAY) == pdTRUE) {
    if (useUCRWiFi) {
      WiFi.disconnect(true);
      WiFi.mode(WIFI_STA);
      WiFi.begin(secureSSID);
      esp_wifi_sta_wpa2_ent_set_identity((uint8_t *)eapIdentity, strlen(eapIdentity));
      esp_wifi_sta_wpa2_ent_set_username((uint8_t *)eapIdentity, strlen(eapIdentity));
      esp_wifi_sta_wpa2_ent_set_password((uint8_t *)eapPassword, strlen(eapPassword));
      esp_wifi_sta_wpa2_ent_enable();
    } else {
      WiFi.disconnect(true);
      WiFi.mode(WIFI_STA);
      WiFi.begin(standardSSID, standardPassword);
    }
    xSemaphoreGive(wifiMutex);
  }
  
  // For LED blink timing
  unsigned long previousWifiBlinkTime = 0;
  unsigned long previousServerBlinkTime = 0;
  bool isBlinkingForServer = false;
  
  // Main loop - connect to server and send data
  char sensorMsg[MSG_BUFFER_SIZE];
  
  while (true) {
    // Reset watchdog
    esp_task_wdt_reset();
    unsigned long currentMillis = millis();
    
    // Check WiFi connection
    if (WiFi.status() != WL_CONNECTED) {
      digitalWrite(wifiLedPin, LOW);
      isBlinkingForServer = false;  // We're now blinking for WiFi
      Serial.println("NetworkTask: WiFi disconnected! Reconnecting...");
      
      if (xSemaphoreTake(wifiMutex, 1000 / portTICK_PERIOD_MS) == pdTRUE) {
        if (useUCRWiFi) {
          WiFi.disconnect(true);
          WiFi.begin(secureSSID);
          esp_wifi_sta_wpa2_ent_set_identity((uint8_t *)eapIdentity, strlen(eapIdentity));
          esp_wifi_sta_wpa2_ent_set_username((uint8_t *)eapIdentity, strlen(eapIdentity));
          esp_wifi_sta_wpa2_ent_set_password((uint8_t *)eapPassword, strlen(eapPassword));
          esp_wifi_sta_wpa2_ent_enable();
        } else {
          WiFi.disconnect(true);
          WiFi.begin(standardSSID, standardPassword);
        }
        xSemaphoreGive(wifiMutex);
      }
      
      // Wait for connection with regular watchdog resets and slow blinking
      unsigned long startTime = millis();
      previousWifiBlinkTime = startTime;
      
      while (WiFi.status() != WL_CONNECTED && millis() - startTime < 10000) {
        esp_task_wdt_reset();
        
        // Handle slow WiFi blinking
        currentMillis = millis();
        if (currentMillis - previousWifiBlinkTime >= WIFI_BLINK_INTERVAL) {
          previousWifiBlinkTime = currentMillis;
          digitalWrite(wifiLedPin, !digitalRead(wifiLedPin));
        }
        
        vTaskDelay(10 / portTICK_PERIOD_MS);
      }
      
      if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\nNetworkTask: WiFi connected!");
        Serial.println("NetworkTask: IP address: " + WiFi.localIP().toString());
        digitalWrite(wifiLedPin, HIGH);
      }
    }
    
    // Connect to server if not connected
    if (!client.connected()) {
      digitalWrite(serverLedPin, LOW);
      isBlinkingForServer = true;  // Now blinking for server
      previousServerBlinkTime = millis();
      Serial.println("NetworkTask: Connecting to server...");
      
      // Set start time for connection attempt
      unsigned long serverConnectStartTime = millis();
      
      // Try to connect with fast blinking for server connection
      while (!client.connected() && millis() - serverConnectStartTime < 5000) {
        // Fast blinking for server connection
        currentMillis = millis();
        if (currentMillis - previousServerBlinkTime >= SERVER_BLINK_INTERVAL) {
          previousServerBlinkTime = currentMillis;
          digitalWrite(serverLedPin, !digitalRead(serverLedPin));
        }
        
        // Periodically try to connect
        if (millis() - serverConnectStartTime > 1000 && 
            millis() - serverConnectStartTime < 1100) {
          client.connect(serverIP, serverPort);
        }
        
        esp_task_wdt_reset();
        vTaskDelay(10 / portTICK_PERIOD_MS);
      }
      
      // Check if we're connected
      if (client.connected()) {
        digitalWrite(serverLedPin, HIGH);
        isBlinkingForServer = false;
        Serial.println("NetworkTask: Connected to server!");
      } else {
        Serial.println("NetworkTask: Failed to connect to server! Retrying...");
        vTaskDelay(1000 / portTICK_PERIOD_MS);
        continue;
      }
    }
    
    // Take queue mutex before accessing the queue
    if (xSemaphoreTake(queueMutex, 10 / portTICK_PERIOD_MS) == pdTRUE) {
      // Check queue and send all available data immediately
      while (xQueueReceive(sensorQueue, sensorMsg, 0) == pdPASS) {
        esp_task_wdt_reset();  // Reset watchdog while processing queue
        
        // Send data to server
        client.println(sensorMsg);
        
        // Look for ACK and potential motor command
        unsigned long startTime = millis();
        bool gotAck = false;
        
        while (millis() - startTime < 100 && !gotAck) {
          if (client.available()) {
            String response = client.readStringUntil('\n');
            
            // Parse command message format "CMD:command"
            if (response.startsWith("CMD:")) {
              String command = response.substring(4); // Extract command after "CMD:"
              command.trim();
              
              // Send motor command via UART
              sendMotorCommand(command.c_str());
              gotAck = true;
            } else if (response == "ACK") {
              // Legacy ACK format
              gotAck = true;
            }
            break;
          }
          vTaskDelay(1);  // Yield to prevent watchdog issues
        }
      }
      // Release queue mutex
      xSemaphoreGive(queueMutex);
    }
    
    // Short delay before checking queue again
    vTaskDelay(1);
  }
}

// ====== TASK 2: SENSOR MEASUREMENT TASK ======
void measurementTask(void *pvParameters) {
  // Register with watchdog
  esp_task_wdt_add(NULL);
  
  // Initialize I2C and MPU6050
  Wire.begin(21, 22);
  Wire.setClock(400000);  // 400kHz I2C clock
  
  mpu.begin(0, 0);
  mpu.setGyroOffsets(gyroXoffset, gyroYoffset, gyroZoffset);
  mpu.setAccOffsets(accXoffset, accYoffset, accZoffset);
  
  Serial.println("MeasurementTask: Starting measurements");
  
  char msg[MSG_BUFFER_SIZE];
  char tempMsg[MSG_BUFFER_SIZE];  // Temporary buffer for queue operations
  
  while (true) {
    // Reset watchdog
    esp_task_wdt_reset();
    
    // Read ultrasonic sensors with short yields between readings
    long leftDist = measureDistance(sigPinLeft);
    vTaskDelay(1);  // Short yield to prevent watchdog issues
    
    long backDist = measureDistance(sigPinBack);
    vTaskDelay(1);
    
    long rightDist = measureDistance(sigPinRight);
    vTaskDelay(1);
    
    long forwardDist = measureDistance(sigPinForward);
    vTaskDelay(1);
    
    // Read MPU6050
    mpu.updateIMU();
    float roll = mpu.getRoll();
    float pitch = mpu.getPitch();
    float yaw = mpu.getYaw();
    
    // Format message
    snprintf(msg, MSG_BUFFER_SIZE,
            "Left: %ld cm, Back: %ld cm, Right: %ld cm, Forward: %ld cm, "
            "Roll: %.2f, Pitch: %.2f, Yaw: %.2f",
            leftDist, backDist, rightDist, forwardDist,
            roll, pitch, yaw);
    
    Serial.println(msg);
    
    // Take queue mutex before attempting to add to the queue
    if (xSemaphoreTake(queueMutex, 10 / portTICK_PERIOD_MS) == pdTRUE) {
      // Try to send to queue - don't block if full
      if (xQueueSend(sensorQueue, &msg, 0) != pdPASS) {
        // Queue is full - remove oldest element and add new one
        if (xQueueReceive(sensorQueue, &tempMsg, 0) == pdPASS) {
          // Successfully removed the oldest item, now add the new one
          if (xQueueSend(sensorQueue, &msg, 0) == pdPASS) {
            Serial.println("Queue full: Replaced oldest measurement with newest one");
          } else {
            Serial.println("Error: Failed to add new measurement after removing old one");
          }
        } else {
          Serial.println("Error: Failed to remove oldest measurement from full queue");
        }
      }
      // Release queue mutex
      xSemaphoreGive(queueMutex);
    } else {
      Serial.println("Warning: Could not acquire queue mutex, skipping measurement");
    }
    
    // Short delay before next measurement
    vTaskDelay(5);
  }
}

// ====== TASK 3: WATCHDOG TASK ======
void watchdogTask(void *pvParameters) {
  // Register with watchdog
  esp_task_wdt_add(NULL);
  
  Serial.println("WatchdogTask: Started");
  
  while (true) {
    // Reset the watchdog timer
    esp_task_wdt_reset();
    
    // Monitor WiFi status for LED indicator - but don't interfere with the blinking patterns
    // This was removed to avoid conflicts with the blinking patterns in the network task
    
    // Short delay
    vTaskDelay(1000 / portTICK_PERIOD_MS);
  }
}

// ====== SETUP ======
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n\n=== ESP32 Sensor Hub Starting ===");
  
  // Initialize UART for motor commands
  motorSerial.begin(9600, SERIAL_8N1, motorRxPin, motorTxPin);
  Serial.println("Motor UART initialized");
  
  // Initialize watchdog timer with longer timeout
  esp_task_wdt_init(WDT_TIMEOUT, true);  // 15-second timeout with panic mode
  
  // Initialize LEDs
  pinMode(wifiLedPin, OUTPUT);
  pinMode(serverLedPin, OUTPUT);
  digitalWrite(wifiLedPin, LOW);
  digitalWrite(serverLedPin, LOW);
  
  // Create mutexes
  wifiMutex = xSemaphoreCreateMutex();
  if (wifiMutex == NULL) {
    Serial.println("Error creating WiFi mutex!");
    while (1) { delay(1000); }
  }
  
  queueMutex = xSemaphoreCreateMutex();
  if (queueMutex == NULL) {
    Serial.println("Error creating queue mutex!");
    while (1) { delay(1000); }
  }
  
  motorMutex = xSemaphoreCreateMutex();
  if (motorMutex == NULL) {
    Serial.println("Error creating motor mutex!");
    while (1) { delay(1000); }
  }
  
  // Create sensor queue with larger size
  sensorQueue = xQueueCreate(SENSOR_QUEUE_SIZE, MSG_BUFFER_SIZE);
  if (sensorQueue == NULL) {
    Serial.println("Error creating sensor queue!");
    while (1) { delay(1000); }
  }
  
  // Create tasks
  xTaskCreatePinnedToCore(
    networkTask,
    "NetworkTask",
    8192,  // Larger stack size for network operations
    NULL,
    2,     // Higher priority
    NULL,
    0      // Run on Core 0
  );
  
  xTaskCreatePinnedToCore(
    measurementTask,
    "MeasurementTask",
    4096,  // Stack size
    NULL,
    1,     // Priority
    NULL,
    1      // Run on Core 1 (separate from network task)
  );
  
  xTaskCreatePinnedToCore(
    watchdogTask,
    "WatchdogTask",
    2048,  // Small stack size
    NULL,
    3,     // Highest priority
    NULL,
    0      // Run on Core 0
  );
  
  // Add the main task to the watchdog
  esp_task_wdt_add(NULL);
  
  Serial.println("All tasks created.");
}

// ====== LOOP ======
void loop() {
  // Reset watchdog in main loop
  esp_task_wdt_reset();
  
  // Main loop is empty since everything happens in tasks
  vTaskDelay(1000 / portTICK_PERIOD_MS);
}