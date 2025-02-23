#include <Arduino.h>
#include <Wire.h>
#include "MPU6050.h"

// Create a TwoWire instance on the default I2C bus (bus 0) of the ESP32
TwoWire myI2C = TwoWire(0);

// Create the MPU6050 object, passing our TwoWire instance
MPU6050 mpu(myI2C);

// We'll track when we last printed, so we can print only every 500 ms
unsigned long lastPrint = 0;
const unsigned long PRINT_INTERVAL = 500; // ms

void setup() {
  Serial.begin(115200);
  delay(100);

  // 1) Initialize I2C on pins SDA=21, SCL=22 at 400kHz
  //    (Adjust pins/frequency if your wiring or board differ)
  myI2C.begin(21, 22);
  myI2C.setClock(1000000);

  // 2) Optional: Quick I2C scan for debugging
  Serial.println("Scanning I2C addresses...");
  for (uint8_t addr = 1; addr < 127; addr++) {
    myI2C.beginTransmission(addr);
    if (myI2C.endTransmission() == 0) {
      Serial.print("Found I2C device at 0x");
      if (addr < 16) Serial.print("0");
      Serial.println(addr, HEX);
    }
  }
  Serial.println("Scan done.\n");

  // 3) Initialize the MPU6050
  //    - 1 => gyro range = ±500 deg/sec
  //    - 0 => accel range = ±2g
  //    (Possible config_num values: 0..3 for each)
  Serial.println("Initializing and calibrating MPU6050...");
  byte status = mpu.begin(/* gyro_config_num = */0, /* acc_config_num = */0);

  // status == 0 means I2C writes were ACKed successfully.
  if (status != 0) {
    Serial.print("MPU6050 connection failed with status: ");
    Serial.println(status);
    // Stop or handle error
    while (true) { delay(100); }
  } else {
    Serial.println("MPU6050 successfully initialized and calibrated.\n");

    // 4) Print the resulting offsets:
    Serial.println("Calculated offsets:");
    Serial.print("  Gyro X offset: "); Serial.println(mpu.getGyroXoffset(), 4);
    Serial.print("  Gyro Y offset: "); Serial.println(mpu.getGyroYoffset(), 4);
    Serial.print("  Gyro Z offset: "); Serial.println(mpu.getGyroZoffset(), 4);

    Serial.print("  Acc  X offset: "); Serial.println(mpu.getAccXoffset(), 4);
    Serial.print("  Acc  Y offset: "); Serial.println(mpu.getAccYoffset(), 4);
    Serial.print("  Acc  Z offset: "); Serial.println(mpu.getAccZoffset(), 4);
    Serial.println();
  }

  // mpu.setGyroOffsets(0.0f, 0.0f, 0.0f);
  // mpu.setAccOffsets(0.0f, 0.0f, 1.0f);
}

void loop() {
  // 1) Update the MPU6050 as fast as possible
  //    - updateIMU() internally calls updateFast() (fetchData + complementary filter)
  //    - Also updates quaternion-based orientation (Madgwick filter)
  mpu.updateIMU();

  // 2) Only print every 500 ms
  unsigned long now = millis();
  if ((now - lastPrint) >= PRINT_INTERVAL) {
    lastPrint = now;

    // Complementary-filter angles:
    float angleX = mpu.getAngleX(); // degrees (X axis)
    float angleY = mpu.getAngleY(); // degrees (Y axis)
    float angleZ = mpu.getAngleZ(); // degrees (Z axis)

    // Madgwick-based Euler angles:
    float roll  = mpu.getRoll();  // degrees
    float pitch = mpu.getPitch(); // degrees
    float yaw   = mpu.getYaw();   // degrees

    // Print everything
    Serial.print("AngleX: "); Serial.print(angleX, 2);
    Serial.print("\tAngleY: "); Serial.print(angleY, 2);
    Serial.print("\tAngleZ: "); Serial.print(angleZ, 2);
    Serial.println();

    Serial.print("Roll: ");  Serial.print(roll, 2);
    Serial.print("\tPitch: "); Serial.print(pitch, 2);
    Serial.print("\tYaw: ");   Serial.println(yaw, 2);

    Serial.println("---------------------------------------------------");
  }
}



// Calculated offsets:
//   Gyro X offset: 0.8279
//   Gyro Y offset: 1.3698
//   Gyro Z offset: -0.0624
//   Acc  X offset: -0.0548
//   Acc  Y offset: -0.0352
//   Acc  Z offset: 0.0515