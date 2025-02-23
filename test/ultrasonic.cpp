#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <esp_wpa2.h>  // Needed for WPA2 Enterprise connections

// ====== FLAG TO CHOOSE NETWORK ======
const bool useUCRWiFi = true; // Set to false to use standard Wi-Fi

// ====== Standard Network Credentials ======
const char* standardSSID = "Imhatin43";
const char* standardPassword = "ILikeToType73";

// ====== WPA2 Enterprise Credentials (e.g., UCR-SECURE) ======
const char* secureSSID = "UCR-SECURE";
const char* eapIdentity = "bmarc018";
const char* eapPassword = "City6464!"; 

// Grove Ultrasonic Ranger Pins (Single Pin for Trig and Echo)
const int sigPinLeft = 18;
const int sigPinBack = 16;
const int sigPinRight = 5;
const int sigPinForward = 17;

// Server details
const char* serverIP = "10.13.192.57";  // Your server's IP address
// const char* serverIP = "192.168.0.33";
const int serverPort = 8642;

// Secure client
WiFiClientSecure client;

// Function to get distance from the single-pin ultrasonic sensor
long getDistance(int sigPin) {
  // Ensure the pin is in output mode for the trigger pulse
  pinMode(sigPin, OUTPUT);

  // Send the trigger pulse:
  digitalWrite(sigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(sigPin, HIGH);
  delayMicroseconds(5); // Minimum ~5us pulse
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
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);  // Always use station mode

  // ====== Choose connection path based on flag ======
  if (useUCRWiFi) {
    // ====== WPA2-Enterprise Connection ======
    Serial.println("Connecting to UCR-SECURE (WPA2-Ent) ...");
    // It is often recommended to disconnect first
    WiFi.disconnect(true);  
    // Begin with SSID (username/password gets set via WPA2 enterprise calls below)
    WiFi.begin(secureSSID);
    
    // Setting up the WPA2 Enterprise environment
    esp_wifi_sta_wpa2_ent_set_identity((uint8_t *)eapIdentity, strlen(eapIdentity));
    // If required, set the username (sometimes same as identity)
    esp_wifi_sta_wpa2_ent_set_username((uint8_t *)eapIdentity, strlen(eapIdentity));
    // Set the WPA2-Ent password
    esp_wifi_sta_wpa2_ent_set_password((uint8_t *)eapPassword, strlen(eapPassword));

    // Enable WPA2-Ent
    esp_wifi_sta_wpa2_ent_enable();
  } else {
    // ====== Standard Wi-Fi Connection ======
    Serial.print("Connecting to standard WiFi: ");
    Serial.println(standardSSID);
    WiFi.disconnect(true);  
    WiFi.mode(WIFI_STA); 
    WiFi.begin(standardSSID, standardPassword);
  }

  // ====== Wait for Wi-Fi to Connect ======
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected!");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());

  // Optional: configure for SSL (using self-signed cert)
  client.setInsecure();  // For testing with self-signed certs ONLY

  // Connect to the server (and attempt reconnection if it fails)
  while (!client.connect(serverIP, serverPort)) {
    Serial.println("Connection to server failed! Retrying...");
    delay(5000);
  }
  Serial.println("Connected to server!");
}

void loop() {
  // Check if the connection is still alive
  if (!client.connected()) {
    Serial.println("Connection lost! Attempting to reconnect...");
    while (!client.connect(serverIP, serverPort)) {
      Serial.println("Reconnection failed! Retrying...");
      delay(5000);
    }
    Serial.println("Reconnected to server!");
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
  Serial.println(dataString); // Also print to the serial monitor

  // Read server response (if any)
  while (client.available()) {
    char c = client.read();
    Serial.print(c);
  }

  delay(25);  // Short delay before next measurement
}
