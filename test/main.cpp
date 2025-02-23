#include <WiFi.h>
#include <WiFiClientSecure.h>
#include "Wire.h"
#include <GY87.h>
#include "esp_wpa2.h"
#include <ESPping.h>

// Wi-Fi Credentials
const char* standardSSID = "Imhatin43";
const char* standardPassword = "ILikeToType73";

const char* secureSSID = "UCR-SECURE";
const char* identity = "bmarc018";
const char* password = "City6464!";

// Choose Wi-Fi mode with flag (set to 1 for UCR-SECURE or 0 for standard)
const bool useUCRWiFi = true;

// Ultrasonic Sensor Pins (Single Pin for Trig and Echo)
const int sigPinLeft = 18;
const int sigPinBack = 16;
const int sigPinRight = 5;
const int sigPinForward = 17;

// GY87 I2C Setup
TwoWire wire = TwoWire(0);  // Use TwoWire instance 0
GY87 gy87(wire); 

// Wi-Fi Client for Secure Connection
WiFiClientSecure client;
uint8_t macAddress[18];

// Google DNS IP for pinging
const char* google_ip = "8.8.8.8";


long getDistance(int sigPin) {
  pinMode(sigPin, OUTPUT);
  digitalWrite(sigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(sigPin, HIGH);
  delayMicroseconds(5);
  digitalWrite(sigPin, LOW);
  pinMode(sigPin, INPUT);
  long duration = pulseIn(sigPin, HIGH);
  return duration * 0.034 / 2;
}

String formatIMUData() {
  String dataString = "";
  dataString += "Temperature: " + String(gy87.getTemp()) + " C\n";
  dataString += "Pressure: " + String(gy87.getPressure()) + " hPa\n";
  dataString += "Altitude: " + String(gy87.getAltitude()) + " m\n";
  dataString += "Position X: " + String(gy87.getPositionX()) + " m\tY: " + String(gy87.getPositionY()) + " m\tZ: " + String(gy87.getPositionZ()) + " m\n";
  dataString += "Velocity X: " + String(gy87.getVelocityX()) + " m/s\tY: " + String(gy87.getVelocityY()) + " m/s\tZ: " + String(gy87.getVelocityZ()) + " m/s\n";
  dataString += "Acceleration X: " + String(gy87.getAccX()) + " m/s²\tY: " + String(gy87.getAccY()) + " m/s²\tZ: " + String(gy87.getAccZ()) + " m/s²\n";
  dataString += "Gyro X: " + String(gy87.getGyroX()) + " dps\tY: " + String(gy87.getGyroY()) + " dps\tZ: " + String(gy87.getGyroZ()) + " dps\n";
  dataString += "Mag X: " + String(gy87.getMagX()) + "\tY: " + String(gy87.getMagY()) + "\tZ: " + String(gy87.getMagZ()) + "\n";
  dataString += "Angle X: " + String(gy87.getAngleX()) + "\tY: " + String(gy87.getAngleY()) + "\tZ: " + String(gy87.getAngleZ()) + " deg\n";
  dataString += "Madgwick Roll: " + String(gy87.getRoll()) + "\tPitch: " + String(gy87.getPitch()) + "\tYaw: " + String(gy87.getYaw()) + " deg\n";
  dataString += "Absolute Yaw: " + String(gy87.getAbsoluteYaw()) + " deg\n";
  return dataString;
}

void setup() {
  Serial.begin(115200);  // Start serial communication
  WiFi.mode(WIFI_STA);  // Set Wi-Fi to STA mode

  // Wi-Fi connection setup
  if (useUCRWiFi) {
    Serial.println("Connecting to UCR-SECURE...");
    WiFi.disconnect(true);  // Disconnect from any previous Wi-Fi
    WiFi.begin(secureSSID);
    esp_wifi_sta_wpa2_ent_set_identity((uint8_t *)identity, strlen(identity));
    esp_wifi_sta_wpa2_ent_set_username((uint8_t *)identity, strlen(identity));
    esp_wifi_sta_wpa2_ent_set_password((uint8_t *)password, strlen(password));
    esp_wifi_sta_wpa2_ent_enable();
  } else {
    Serial.println("Connecting to standard Wi-Fi...");
    WiFi.begin(standardSSID, standardPassword);
  }

  // Wait for Wi-Fi to connect
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWi-Fi connected");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  // Initialize the Wi-Fi Client
  WiFi.macAddress(macAddress);
  client.setInsecure();  // For testing with self-signed certs ONLY

  // Connect to the server
  const char* serverIP = "10.13.171.70";  // Change to your server IP
  const int serverPort = 8642;
  while (!client.connect(serverIP, serverPort)) {
    Serial.println("Connection failed! Retrying...");
    delay(1000);
  }
  Serial.println("Connected to server!");

  // Initialize I2C for GY87 sensor
  wire.begin(23, 22);
  gy87.begin();
}

void loop() {
  // --- Ultrasonic Sensor Data ---
  long distanceLeft = getDistance(sigPinLeft);
  long distanceBack = getDistance(sigPinBack);
  long distanceRight = getDistance(sigPinRight);
  long distanceForward = getDistance(sigPinForward);

  String ultrasonicDataString = "Left: " + String(distanceLeft) + " cm, ";
  ultrasonicDataString += "Back: " + String(distanceBack) + " cm, ";
  ultrasonicDataString += "Right: " + String(distanceRight) + " cm, ";
  ultrasonicDataString += "Forward: " + String(distanceForward) + " cm\n";

  // --- GY87 IMU Data ---
  String imuDataString = formatIMUData();

  // --- Combine Data ---
  String combinedDataString = ultrasonicDataString + imuDataString;

  // Print data to Serial
  Serial.println(combinedDataString);

  // Send data over Wi-Fi
  client.println(combinedDataString);

  // Wait before taking new measurements
  delay(200);
}
