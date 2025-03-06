#include <HardwareSerial.h>

// Define hardware serial (Serial2)
HardwareSerial motorSerial(2);  // Using UART2 on ESP32

// UART pins
const int motorRxPin = 19;  // RX pin connected to receiver's TX
const int motorTxPin = 23;  // TX pin connected to receiver's RX

void sendCommand(const char* command) {
  motorSerial.print(command);
  motorSerial.print("\n");  // Send newline to terminate command
  Serial.println("Sent: " + String(command));
}

void checkResponse() {
  unsigned long startTime = millis();
  while (!motorSerial.available() && millis() - startTime < 1000) {
    delay(10);  // Wait up to 1 second for data to arrive
  }
  if (motorSerial.available()) {
    String response = motorSerial.readStringUntil('\n');
    response.trim();  // Remove any trailing \r or whitespace
    Serial.println("Received: " + response);
  } else {
    Serial.println("No response received");
  }
}

void setup() {
  Serial.begin(115200);  // Debug serial
  delay(1000);
  Serial.println("=== UART Test Starting ===");
  
  motorSerial.begin(9600, SERIAL_8N1, motorRxPin, motorTxPin);
  Serial.println("Motor UART initialized with HardwareSerial");
  
  delay(2000);  // Give the receiver time to initialize
  sendCommand("forward");
  delay(500);   // Wait 500ms before checking response
  checkResponse();
}

void loop() {
  static int state = 0;
  static unsigned long lastChange = 0;
  
  if (millis() - lastChange > 5000) {  // Send command every 5 seconds
    lastChange = millis();
    
    switch(state) {
      case 0:
        sendCommand("forward");
        break;
      case 1:
        sendCommand("left");
        break;
      case 2:
        sendCommand("right");
        break;
      case 3:
        sendCommand("reverse");
        break;
      default:
        sendCommand("stop");
        state = -1;  // Reset state after stop
        break;
    }
    
    state++;
    
    delay(500);  // Wait 500ms before checking response
    checkResponse();
  }
}