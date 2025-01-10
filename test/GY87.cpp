#include "Wire.h"
#include <GY87.h>
#include <SPI.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>

TwoWire wire = TwoWire(0);  // Use TwoWire instance 0
GY87 gy87(wire);

const char* ssid = "Imhatin43";
const char* password = "ILikeToType73";
const char* serverIP = "192.168.0.80";  // Your server's IP address
const int serverPort = 8080;

WiFiClientSecure client;

String formatIMUData() {
  String dataString = "";
  dataString += "TEMPERATURE: " + String(gy87.getTemp());
  dataString += "\nPRESSURE: " + String(gy87.getPressure());
  dataString += "\nALTITUDE: " + String(gy87.getAltitude());
  dataString += "\nPOS       X: " + String(gy87.getPositionX()) + "\tY: " + String(gy87.getPositionY()) + "\tZ: " + String(gy87.getPositionZ());
  dataString += "\nVEL       X: " + String(gy87.getVelocityX()) + "\tY: " + String(gy87.getVelocityY()) + "\tZ: " + String(gy87.getVelocityZ());
  dataString += "\nACCEL     X: " + String(gy87.getAccX()) + "\tY: " + String(gy87.getAccY()) + "\tZ: " + String(gy87.getAccZ());
  dataString += "\nGYRO      X: " + String(gy87.getGyroX()) + "\tY: " + String(gy87.getGyroY()) + "\tZ: " + String(gy87.getGyroZ());
  dataString += "\nMAG       X: " + String(gy87.getMagX()) + "\tY: " + String(gy87.getMagY()) + "\tZ: " + String(gy87.getMagZ());
  dataString += "\nANGLE     X: " + String(gy87.getAngleX()) + "\tY: " + String(gy87.getAngleY()) + "\tZ: " + String(gy87.getAngleZ());
  dataString += "\nACC ANGLE X: " + String(gy87.getAccAngleX()) + "\tY: " + String(gy87.getAccAngleY());
  dataString += "\nMADGWICK Roll: " + String(gy87.getRoll()) + "\tPitch: " + String(gy87.getPitch()) + "\tYaw: " + String(gy87.getYaw());
  dataString += "\nAbsolute Yaw: " + String(gy87.getAbsoluteYaw());
  dataString += "\n"; // Empty line as separator - important for the server
  return dataString;
}

void setup() {
  // No Serial.begin()!

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }

  client.setInsecure();

  // Initialize I2C on custom pins
  wire.begin(1, 3);  // SDA = GPIO 1, SCL = GPIO 3
  pinMode(1, INPUT_PULLUP);
  pinMode(3, INPUT_PULLUP);

  if (gy87.begin(3, 3, 3, true) != 0) {
    while(1); // Stop here if calibration fails
  }

  delay(1000);
  gy87.setAccOffsets(-0.05, -0.01, 0.03);
  gy87.setGyroOffsets(-1.10, 3.25, -0.79);
  gy87.stabilize(true, 1000);

  // Connect to the server
  while (!client.connect(serverIP, serverPort)) {
    delay(5000);
  }
}

void loop() {
  if (!client.connected()) {
    while (!client.connect(serverIP, serverPort)) {
      delay(5000);
    }
  }

  gy87.updateIMU();

  // Get all IMU data 
  gy87.getTemp();
  gy87.getPressure();
  gy87.getAltitude();
  gy87.getAccX();
  gy87.getAccY();
  gy87.getAccZ();
  gy87.getGyroX();
  gy87.getGyroY();
  gy87.getGyroZ();
  gy87.getMagX();
  gy87.getMagY();
  gy87.getMagZ();
  gy87.getAccAngleX();
  gy87.getAccAngleY();
  gy87.getAngleX();
  gy87.getAngleY();
  gy87.getAngleZ();
  gy87.getRoll();
  gy87.getPitch();
  gy87.getYaw();
  gy87.getPositionX();
  gy87.getPositionY();
  gy87.getPositionZ();
  gy87.getVelocityX();
  gy87.getVelocityY();
  gy87.getVelocityZ();

  // Format data into a string
  String dataString = formatIMUData();

  // Send data to the server
  client.println(dataString);

  // No response reading needed (or you can add it back if you want)

  delay(100); // Adjust delay as needed
}