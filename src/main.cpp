#include <WiFi.h>
#include "esp_wpa2.h"
#include <ESPping.h>  // Include the ESPping library

const char* ssid = "UCR-SECURE";
const char* identity = "";
const char* password = "";

// Google DNS IP for pinging
const char* google_ip = "8.8.8.8";

void setup() {
  Serial.begin(115200);
  WiFi.disconnect(true);  // Disconnect from any previous WiFi connection
  WiFi.mode(WIFI_STA);

  // Configure WPA2 Enterprise
  esp_wifi_sta_wpa2_ent_set_identity((uint8_t *)identity, strlen(identity));
  esp_wifi_sta_wpa2_ent_set_username((uint8_t *)identity, strlen(identity));  // For MSCHAPv2, identity == username
  esp_wifi_sta_wpa2_ent_set_password((uint8_t *)password, strlen(password));

  // Clear the CA cert if not needed
  esp_wifi_sta_wpa2_ent_set_ca_cert(NULL, 0);

  // Enable WPA2 Enterprise
  esp_wifi_sta_wpa2_ent_enable();

  // Start the WiFi connection
  WiFi.begin(ssid); 

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  // Begin ping test to Google DNS
  Serial.println("Pinging Google DNS (8.8.8.8)...");
}

void loop() {
  // Ping Google DNS (8.8.8.8) and wait for a result
  bool success = Ping.ping(google_ip, 4);  // Send 4 pings

  // Check the result
  if (success) {
    Serial.println("Ping successful!");

    // Print statistics after pinging
    Serial.print("Average time: ");
    Serial.print(Ping.averageTime());
    Serial.println(" ms");

    Serial.print("Min time: ");
    Serial.print(Ping.minTime());
    Serial.println(" ms");

    Serial.print("Max time: ");
    Serial.print(Ping.maxTime());
    Serial.println(" ms");
  } else {
    Serial.println("Ping failed!");
  }

  delay(5000);  // Wait 5 seconds before the next ping
}
