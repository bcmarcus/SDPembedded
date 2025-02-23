#include <WiFi.h>
#include "Wire.h"
#include <GY87.h>
#include <SPI.h>

TwoWire wire = TwoWire(0);  // Use TwoWire instance 0
GY87 gy87(wire);

long timer = 0;
int counter = 0;

void printData() {
  Serial.print(F("TEMPERATURE: "));Serial.println(gy87.getTemp());
  Serial.print(F("PRESSURE: "));Serial.println(gy87.getPressure());
  Serial.print(F("ALTITUDE: "));Serial.println(gy87.getAltitude());

  Serial.print(F("GYRO      X: "));Serial.print(gy87.getGyroX());
  Serial.print("\tY: ");Serial.print(gy87.getGyroY());
  Serial.print("\tZ: ");Serial.println(gy87.getGyroZ());
  
  Serial.print(F("MAG       X: "));Serial.print(gy87.getMagX());
  Serial.print("\tY: ");Serial.print(gy87.getMagY());
  Serial.print("\tZ: ");Serial.println(gy87.getMagZ());
  
  Serial.print(F("ANGLE     X: "));Serial.print(gy87.getAngleX());
  Serial.print("\tY: ");Serial.print(gy87.getAngleY());
  Serial.print("\tZ: ");Serial.println(gy87.getAngleZ());

  Serial.print(F("ACC ANGLE X: "));Serial.print(gy87.getAccAngleX());
  Serial.print("\tY: ");Serial.println(gy87.getAccAngleY());

  Serial.print(F("MADGWICK Roll: "));Serial.print(gy87.getRoll());
  Serial.print("\tPitch: ");Serial.print(gy87.getPitch());
  Serial.print("\tYaw: ");Serial.println(gy87.getYaw());
  Serial.print("\tAbsolute Yaw: ");Serial.println(gy87.getAbsoluteYaw());
  Serial.println(F("=====================================================\n"));
}


void setup() {
  Serial.begin(115200);

  Serial.println ("Starting");
  // Initialize I2C on custom pins
  // wire.begin(23, 22);  // SDA = GPIO 23, SCL = GPIO 22
  wire.begin();  // 100kHz clock speed
  wire.setClock(100000);
  wire.setBufferSize(256);

  for (uint8_t address = 1; address < 127; address++) {
    wire.beginTransmission(address);
    uint8_t error = wire.endTransmission();

    if (error == 0) {
      Serial.print("I2C device found at address 0x");
      if (address < 16) Serial.print("0"); // Leading zero for addresses < 0x10
      Serial.println(address, HEX);
    }
  }

  // Print a newline to separate scans and restart immediately
  Serial.println("Scan complete.\n");



  // // Enable internal pull-up resistors on the custom SDA and SCL pins
  // pinMode(23, INPUT_PULLUP); // SDA (GPIO 23)
  // pinMode(22, INPUT_PULLUP); // SCL (GPIO 22)

  Serial.println(F("Calibrating sensor, do not move GY87"));
  if (gy87.begin(3, 3, 3, true) != 0) {
    Serial.println("Calibration failed.");
    // delay (10000);
  }
  Serial.println(F("Calibration succeeded, starting test."));
  
  Serial.print(F("ACCEL OFFSETS     X: "));Serial.print(gy87.getAccXoffset());
  Serial.print("\tY: ");Serial.print(gy87.getAccYoffset());
  Serial.print("\tZ: ");Serial.println(gy87.getAccZoffset());
  
  Serial.print(F("GYRO OFFSETS     X: "));Serial.print(gy87.getGyroXoffset());
  Serial.print("\tY: ");Serial.print(gy87.getGyroYoffset());
  Serial.print("\tZ: ");Serial.println(gy87.getGyroZoffset());

  delay(1000);
  gy87.setAccOffsets(-0.05, -0.01, 0.03);
  gy87.setGyroOffsets(-1.10, 3.25, -0.79);
  gy87.stabilize(true, 1000);
}

void loop() {
  Serial.println ("Looping");
  gy87.updateIMU();


  counter++;

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



  // COMMENT OUT TO TEST SPEED //

  printData();
  delay (100);

  // COMMENT OUT TO TEST SPEED //

  if (counter % 10000 == 0) {
    Serial.print(counter);
    Serial.print(" :: ");
    Serial.print(millis() / 1000.0);
    Serial.print(" :: ");
    Serial.println((millis() - timer) / 1000.0);
    timer = millis();
    printData();
  }
}