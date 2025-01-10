#include <WiFi.h>
#include <WiFiClientSecure.h>

// Grove Ultrasonic Ranger Pins (Single Pin for Trig and Echo)
const int sigPinLeft = 18;
const int sigPinBack = 16;
const int sigPinRight = 5;   // Example: Replace with your actual pin
const int sigPinForward = 17; // Example: Replace with your actual pin

const char* ssid = "Imhatin43";
const char* password = "ILikeToType73";
const char* serverIP = "192.168.0.80";  // Your server's IP address
const int serverPort = 8080;

WiFiClientSecure client;

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

void setup() {
  // Serial.begin(115200);

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  // Serial.println("\nWiFi connected");
  // Serial.println("IP address: ");
  // Serial.println(WiFi.localIP());

  // Optional: Configure for SSL (using self-signed cert)
  client.setInsecure();  // For testing with self-signed certs ONLY

  // Connect to the server (and attempt reconnection if it fails)
  while (!client.connect(serverIP, serverPort)) {
    Serial.println("Connection failed! Retrying...");
    delay(5000);
  }
  Serial.println("Connected to server!");
}

void loop() {
  // Check if the connection is still alive
  if (!client.connected()) {
    // Serial.println("Connection lost! Attempting to reconnect...");
    while (!client.connect(serverIP, serverPort)) {
      // Serial.println("Reconnection failed! Retrying...");
      delay(5000);
    }
    // Serial.println("Reconnected to server!");
  }

  // Get distance readings
  long distanceLeft = getDistance(sigPinLeft);
  long distanceBack = getDistance(sigPinBack);
  long distanceRight = getDistance(sigPinRight);
  long distanceForward = getDistance(sigPinForward);

  // Create a string to send to the server
  String dataString = "Left: " + String(distanceLeft) + " cm, ";
  dataString += "Back: " + String(distanceBack) + " cm, ";
  dataString += "Right: " + String(distanceRight) + " cm, ";
  dataString += "Forward: " + String(distanceForward) + " cm";

  // Send data to the server
  client.println(dataString);
  // Serial.println(dataString); // Also print to the serial monitor

  // Read response (if any) - Non-blocking
  while (client.available()) {
    char c = client.read();
    // Serial.print(c);
  }

  delay(20);  // Wait 1 second before taking new measurements
}