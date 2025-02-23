#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <esp_wpa2.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include <Wire.h>
#include "MPU6050.h"

// ====== NETWORK CONFIGURATION ======
const bool useUCRWiFi = false; // true: use WPA2-Enterprise; false: use standard Wi-Fi

// Standard Wi-Fi credentials
const char* standardSSID     = "Imhatin43";
const char* standardPassword = "ILikeToType73";

// WPA2 Enterprise credentials  (if using UCR-SECURE)
const char* secureSSID     = "UCR-SECURE";
const char* eapIdentity    = "bmarc018";
const char* eapPassword    = "City6464!";

// ====== SENSOR PIN ASSIGNMENTS ======
const int sigPinLeft    = 18;
const int sigPinBack    = 16;
const int sigPinRight   = 5;
const int sigPinForward = 17;

// ====== SERVER CONFIGURATION ======
const char* serverIP   = "192.168.0.152";  // Replace with your server's IP
const int   serverPort = 8642;

// ====== QUEUE SETTINGS ======
const int SENSOR_QUEUE_SIZE = 100;      // Maximum number of messages queued
const int MSG_BUFFER_SIZE   = 128;      // Size of each message in bytes

QueueHandle_t sensorQueue = NULL;
SemaphoreHandle_t serverConnectedSemaphore = NULL;

// ====== MPU6050 CONFIGURATION ======
MPU6050 mpu(Wire);
float gyroXoffset = 0.8279;
float gyroYoffset = 1.3698;
float gyroZoffset = -0.0624;
float accXoffset = -0.0548;
float accYoffset = -0.0352;
float accZoffset = 0.0515;

// ====== HELPER FUNCTION: Measure Distance ======
long getDistance(int sigPin) {
  pinMode(sigPin, OUTPUT);
  digitalWrite(sigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(sigPin, HIGH);
  delayMicroseconds(5);  // Minimum ~5µs pulse
  digitalWrite(sigPin, LOW);
  
  pinMode(sigPin, INPUT);
  long duration = pulseIn(sigPin, HIGH, 30000); // timeout: 30 ms
  long distance = duration * 0.034 / 2;        // speed of sound ~0.034 cm/µs
  return distance;
}

// ====== TASK 1: Sensor Measurement (every 25 ms) ======
void measurementTask(void *pvParameters) {
  // Wait for the server connection to be established
  xSemaphoreTake(serverConnectedSemaphore, portMAX_DELAY);
  Serial.println("MeasurementTask: Server connected, starting measurements.");

  // Initialize MPU6050 and set the offsets
  Wire.begin(21, 22);
  mpu.begin(0, 0);  // Initialize the MPU6050
  mpu.setGyroOffsets(gyroXoffset, gyroYoffset, gyroZoffset);  // Set gyro offsets
  mpu.setAccOffsets(accXoffset, accYoffset, accZoffset);  // Set accelerometer offsets

  char msg[MSG_BUFFER_SIZE];

  while (true) {
    // Read ultrasonic sensor data
    long distanceLeft    = getDistance(sigPinLeft);
    long distanceBack    = getDistance(sigPinBack);
    long distanceRight   = getDistance(sigPinRight);
    long distanceForward = getDistance(sigPinForward);

    // Read MPU6050 data
    mpu.updateIMU();
    float roll = mpu.getRoll();
    float pitch = mpu.getPitch();
    float yaw = mpu.getYaw();

    // Format the sensor readings into a fixed-size string
    snprintf(msg, MSG_BUFFER_SIZE,
             "Left: %ld cm, Back: %ld cm, Right: %ld cm, Forward: %ld cm, "
             "Roll: %.2f, Pitch: %.2f, Yaw: %.2f",
             distanceLeft, distanceBack, distanceRight, distanceForward,
             roll, pitch, yaw);

    // For debugging, print to Serial
    Serial.println(msg);

    // Enqueue the sensor reading (if full, drop the message)
    if (xQueueSend(sensorQueue, &msg, 0) != pdPASS) {
      Serial.println("Warning: sensor queue full, dropping measurement.");
    }

    // Wait 25 ms before the next measurement
    vTaskDelay(pdMS_TO_TICKS(25));
  }
}

// ====== TASK 2: Network Transmission ======
void networkTask(void *pvParameters) {
  WiFiClientSecure client;
  client.setInsecure();  // Use with self-signed certificates ONLY

  // Initial connection to the server
  Serial.println("NetworkTask: Connecting to server...");
  while (!client.connect(serverIP, serverPort)) {
    Serial.println("NetworkTask: Connection to server failed! Retrying in 1 sec...");
    vTaskDelay(pdMS_TO_TICKS(1000));
  }
  Serial.println("NetworkTask: Connected to server!");

  // Signal that the server is connected
  xSemaphoreGive(serverConnectedSemaphore);

  char sensorMsg[MSG_BUFFER_SIZE];  // Buffer to hold received messages from the queue

  // Main loop: Send data as it becomes available
  while (true) {
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("NetworkTask: Wi-Fi disconnected! Reconnecting...");
      while (WiFi.status() != WL_CONNECTED) {
        vTaskDelay(pdMS_TO_TICKS(500));
      }
      Serial.println("NetworkTask: Wi-Fi reconnected.");
    }
    if (!client.connected()) {
      Serial.println("NetworkTask: Server connection lost! Reconnecting...");
      while (!client.connect(serverIP, serverPort)) {
        Serial.println("NetworkTask: Reconnecting to server...");
        vTaskDelay(pdMS_TO_TICKS(5000));
      }
      Serial.println("NetworkTask: Reconnected to server!");
    }
    
    if (xQueueReceive(sensorQueue, sensorMsg, pdMS_TO_TICKS(10)) == pdPASS) {
      client.println(sensorMsg);
    } else {
      vTaskDelay(pdMS_TO_TICKS(5));
    }
  }
}

// ====== SETUP & MAIN LOOP ======
void setup() {
  Serial.begin(115200);
  delay(1000);

  // Connect to Wi-Fi
  Serial.println("Setup: Connecting to Wi-Fi...");
  if (useUCRWiFi) {
    WiFi.disconnect(true);
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
  
  // Wait for Wi-Fi to connect
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    vTaskDelay(pdMS_TO_TICKS(500));
  }
  Serial.println("\nSetup: Wi-Fi connected!");
  Serial.println("Setup: IP address: " + WiFi.localIP().toString());

  // Create the sensor queue
  sensorQueue = xQueueCreate(SENSOR_QUEUE_SIZE, MSG_BUFFER_SIZE);
  if (sensorQueue == NULL) {
    Serial.println("Error creating sensor queue!");
    while (true) { vTaskDelay(pdMS_TO_TICKS(1000)); }
  }

  // Create the binary semaphore
  serverConnectedSemaphore = xSemaphoreCreateBinary();
  if (serverConnectedSemaphore == NULL) {
    Serial.println("Error creating serverConnectedSemaphore!");
    while (true) { vTaskDelay(pdMS_TO_TICKS(1000)); }
  }

  // Create tasks
  xTaskCreate(measurementTask, "MeasurementTask", 4096, NULL, 2, NULL);
  xTaskCreate(networkTask, "NetworkTask", 8192, NULL, 1, NULL);
}

void loop() {
  // Empty as tasks are running
  vTaskDelay(pdMS_TO_TICKS(1000));
}