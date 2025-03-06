#include <Arduino.h>
#include <iostream>
#include <HardwareSerial.h>

//Rear Motors
const int ena1 = 12;
const int M1P1 = 13;
const int M1P2 = 14;
const int M2P1 = 27;
const int M2P2 = 26;
const int enb1 = 15;

//Front Motors
const int ena2 = 25;
const int M3P1 = 33;
const int M3P2 = 32;
const int M4P1 = 18;
const int M4P2 = 19;
const int enb2 = 4;

// UART pins for Serial2
const int rxPin = 16;  // Green wire is rx
const int txPin = 17;  // Blue wire is tx

String state = "";

// Function to send feedback over UART
void sendFeedback(String message) {
  Serial2.println(message);
  Serial.println("Feedback sent: " + message);
}

void setup() {
  Serial.begin(115200);  // Debug serial
  Serial2.begin(9600, SERIAL_8N1, rxPin, txPin);  // Communication serial
  
  // Initialize motor pins
  pinMode(ena1, OUTPUT);
  pinMode(M1P1, OUTPUT);
  pinMode(M1P2, OUTPUT);
  pinMode(M2P1, OUTPUT);
  pinMode(M2P2, OUTPUT);
  pinMode(enb1, OUTPUT);
  pinMode(ena2, OUTPUT);
  pinMode(M3P1, OUTPUT);
  pinMode(M3P2, OUTPUT);
  pinMode(M4P1, OUTPUT);
  pinMode(M4P2, OUTPUT);
  pinMode(enb2, OUTPUT); 
  
  Serial.println("Motor controller initialized");
  sendFeedback("ready");
}

void loop() {
  if (Serial2.available()) {
    state = Serial2.readStringUntil('\n'); // Read data
    state.trim();
   
    Serial.println("Received command: " + state);
    
    // Echo back the command as feedback
    sendFeedback(state);

    if (state == "forward") {
        Serial.println("Moving forward");
        digitalWrite(M1P1, LOW);
        digitalWrite(M1P2, HIGH);
        digitalWrite(M2P1, LOW);
        digitalWrite(M2P2, HIGH);
        analogWrite(ena1, 100);
        analogWrite(enb1, 100);
        digitalWrite(M3P1, HIGH);
        digitalWrite(M3P2, LOW);
        digitalWrite(M4P1, HIGH);
        digitalWrite(M4P2, LOW);
        analogWrite(ena2, 100);
        analogWrite(enb2, 100);
    }
    else if (state == "reverse") {
        Serial.println("Moving reverse");
        digitalWrite(M1P1, HIGH);
        digitalWrite(M1P2, LOW);
        digitalWrite(M2P1, HIGH);
        digitalWrite(M2P2, LOW);
        analogWrite(ena1, 100);
        analogWrite(enb1, 100);
        digitalWrite(M3P1, LOW);
        digitalWrite(M3P2, HIGH);
        digitalWrite(M4P1, LOW);
        digitalWrite(M4P2, HIGH);
        analogWrite(ena2, 100);
        analogWrite(enb2, 100);
    }
    else if (state == "right") {
        Serial.println("Turning right");
        digitalWrite(M1P1, HIGH);
        digitalWrite(M1P2, LOW);
        digitalWrite(M2P1, LOW);
        digitalWrite(M2P2, HIGH);
        analogWrite(ena1, 100);
        analogWrite(enb1, 100);
        digitalWrite(M3P1, HIGH);
        digitalWrite(M3P2, LOW);
        digitalWrite(M4P1, LOW);
        digitalWrite(M4P2, HIGH);
        analogWrite(ena2, 100);
        analogWrite(enb2, 100);
    }
    else if (state == "left") {
        Serial.println("Turning left");
        digitalWrite(M1P1, LOW);
        digitalWrite(M1P2, HIGH);
        digitalWrite(M2P1, HIGH);
        digitalWrite(M2P2, LOW);
        analogWrite(ena1, 100);
        analogWrite(enb1, 100);
        digitalWrite(M3P1, LOW);
        digitalWrite(M3P2, HIGH);
        digitalWrite(M4P1, HIGH);
        digitalWrite(M4P2, LOW);
        analogWrite(ena2, 100);
        analogWrite(enb2, 100);
    }
    else if (state == "stop") {
        Serial.println("Stopping");
        digitalWrite(M1P1, LOW);
        digitalWrite(M1P2, LOW);
        digitalWrite(M2P1, LOW);
        digitalWrite(M2P2, LOW);
        analogWrite(ena1, 0);
        analogWrite(enb1, 0);
        digitalWrite(M3P1, LOW);
        digitalWrite(M3P2, LOW);
        digitalWrite(M4P1, LOW);
        digitalWrite(M4P2, LOW);
        analogWrite(ena2, 0);
        analogWrite(enb2, 0);
    }
    else {
        Serial.println("Unknown command, stopping");
        digitalWrite(M1P1, LOW);
        digitalWrite(M1P2, LOW);
        digitalWrite(M2P1, LOW);
        digitalWrite(M2P2, LOW);
        analogWrite(ena1, 0);
        analogWrite(enb1, 0);
        digitalWrite(M3P1, LOW);
        digitalWrite(M3P2, LOW);
        digitalWrite(M4P1, LOW);
        digitalWrite(M4P2, LOW);
        analogWrite(ena2, 0);
        analogWrite(enb2, 0);
        
        // Send error feedback for unknown command
        sendFeedback("error: unknown command");
    }
  } else {
    if (millis() % 1000 == 0) {
      Serial.println("No command received");
    }
  }
}