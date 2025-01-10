#include <WiFi.h>
#include <WiFiClientSecure.h>
#include "Wire.h"
#include <GY87.h>

// Grove Ultrasonic Ranger Pins (Single Pin for Trig and Echo)
const int sigPinLeft = 18;
const int sigPinBack = 16;
const int sigPinRight = 5;   // Example: Replace with your actual pin
const int sigPinForward = 17; // Example: Replace with your actual pin

// GY87 I2C Pins
const int sdaPin = 1;  // SDA
const int sclPin = 3;  // SCL

const char* ssid = "Imhatin43";
const char* password = "ILikeToType73";
// const char* serverIP = "192.168.0.80";  // Your server's IP address
const char* serverIP = "127.0.0.1";
const int serverPort = 8642;

uint8_t macAddress[18];  

WiFiClientSecure client;
TwoWire wire = TwoWire(0);  // Use TwoWire instance 0
GY87 gy87(wire);

void debugPrint(String message) {
  // client.println(String(macAddress) + ": " + message);
}

// Function to get distance from the single-pin ultrasonic sensor
long getDistance(int sigPin) {
  // Ensure the pin is in output mode for the trigger pulse
  pinMode(sigPin, OUTPUT);

  // Send the trigger pulse:
  digitalWrite(sigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(sigPin, HIGH);
  delayMicroseconds(5); // Minimum 5us pulse
  digitalWrite(sigPin, LOW);

  // Switch the pin to input mode to measure the echo
  pinMode(sigPin, INPUT);

  // Measure the echo pulse duration
  long duration = pulseIn(sigPin, HIGH);

  // Calculate the distance (speed of sound = 340 m/s or 0.034 cm/us)
  long distance = duration * 0.034 / 2;

  return distance;
}

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
  dataString += "\n";
  return dataString;
}

void setup() {
  // Serial.begin(115200); // Optional: For debugging

  // Connect to Wi-Fi
  WiFi.mode(WIFI_STA); 
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    // Serial.print(".");
  }

  WiFi.macAddress(macAddress); 

  // Optional: Configure for SSL (using self-signed cert)
  client.setInsecure();  // For testing with self-signed certs ONLY

  // Connect to the server (and attempt reconnection if it fails)
  while (!client.connect(serverIP, serverPort)) {
    // Serial.println("Connection failed! Retrying...");
    delay(1000);
  }
  client.println("Connected to server!");
  delay (100);
  client.println((char)macAddress[0]);
  delay (10000);
  // debugPrint("Connected to server!");

  // Initialize I2C on custom pins for GY87
  pinMode(sdaPin, INPUT_PULLUP);
  pinMode(sclPin, INPUT_PULLUP);
  wire.begin(sdaPin, sclPin, 8000000);

  client.println("Wire initialized!");

  // if (gy87.begin(3, 3, 3, true) != 0) {
    // Serial.println("GY87 initialization failed!");
    // while (1); // Stop here if calibration fails
    // delay (1);
  // }

  // delay(1000);
  // gy87.setAccOffsets(-0.05, -0.01, 0.03);
  // gy87.setGyroOffsets(-1.10, 3.25, -0.79);
  // gy87.stabilize(true, 1000);
}

void loop() {
  // Check if the connection is still alive
  if (!client.connected()) {
    // Serial.println("Connection lost! Attempting to reconnect...");
    while (!client.connect(serverIP, serverPort)) {
      // Serial.println("Reconnection failed! Retrying...");
      delay(1000);
    }
    // Serial.println("Reconnected to server!");
  }

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
  // gy87.updateIMU();

  // Get all IMU data (you don't need to call each get function individually if you've already called updateIMU)
  // String imuDataString = formatIMUData();

  // --- Combine and Send Data ---
  // String combinedDataString = ultrasonicDataString + imuDataString;
  // client.println(combinedDataString);
  client.println(ultrasonicDataString);

  // Serial.println(combinedDataString); // Optional: Print to serial monitor

  // Read response (if any) - Non-blocking
  while (client.available()) {
    char c = client.read();
    // Serial.print(c);
  }

  delay(200);  // Wait before taking new measurements (adjust as needed)
}