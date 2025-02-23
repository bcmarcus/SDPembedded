#include <Wire.h>
#include <Arduino.h>

#define MPU6050_ADDR 0x68    // I2C address for MPU6050 (assuming AD0 pin is grounded)
#define WHO_AM_I      0x75   // MPU6050 WHO_AM_I register
#define PWR_MGMT_1    0x6B   // MPU6050 power management register
#define ACCEL_XOUT_H  0x3B   // Start of accelerometer data registers

// Create a TwoWire instance for I2C bus 0 (the default on ESP32)
TwoWire I2C_0 = TwoWire(0);

// Raw accelerometer readings
int16_t accelX, accelY, accelZ;

// Helper function to print bits of a byte
void printBits(byte b) {
  for (int i = 7; i >= 0; i--) {
    Serial.print(bitRead(b, i));
  }
}

// Helper function to print bits of a 16-bit integer
void printBits(int16_t val) {
    for (int i = 15; i >= 0; i--) {
        Serial.print(bitRead(val, i));
    }
}

/**
 * Reads the 6 bytes of accelerometer data starting at ACCEL_XOUT_H.
 * Updates the global variables accelX, accelY, accelZ.
 */
void readAccelData() {
  I2C_0.beginTransmission(MPU6050_ADDR);
  I2C_0.write(ACCEL_XOUT_H);
  I2C_0.endTransmission(false);        // Send repeated start
  I2C_0.requestFrom(MPU6050_ADDR, 6, true);

  // Read high and low bytes for each axis
  byte accelX_H = I2C_0.read();
  byte accelX_L = I2C_0.read();
  accelX = (accelX_H << 8) | accelX_L;  // X-axis

  byte accelY_H = I2C_0.read();
  byte accelY_L = I2C_0.read();
  accelY = (accelY_H << 8) | accelY_L;  // Y-axis

  byte accelZ_H = I2C_0.read();
  byte accelZ_L = I2C_0.read();
  accelZ = (accelZ_H << 8) | accelZ_L;  // Z-axis

  // Print bits for each byte and the combined value
  Serial.println("Accelerometer Data (Bits):");

  Serial.print("AccelX_H: ");
  printBits(accelX_H);
  Serial.print(" AccelX_L: ");
  printBits(accelX_L);
  Serial.print(" AccelX: ");
  printBits(accelX);
  Serial.println();

  Serial.print("AccelY_H: ");
  printBits(accelY_H);
  Serial.print(" AccelY_L: ");
  printBits(accelY_L);
  Serial.print(" AccelY: ");
  printBits(accelY);
  Serial.println();

  Serial.print("AccelZ_H: ");
  printBits(accelZ_H);
  Serial.print(" AccelZ_L: ");
  printBits(accelZ_L);
  Serial.print(" AccelZ: ");
  printBits(accelZ);
  Serial.println();
}

/**
 * Helper function to write a single byte to a register on the MPU6050.
 */
void writeRegister(uint8_t reg, uint8_t data) {
  I2C_0.beginTransmission(MPU6050_ADDR);
  I2C_0.write(reg);
  I2C_0.write(data);
  I2C_0.endTransmission(true);
}

/**
 * Helper function to read a single byte from a register on the MPU6050.
 */
uint8_t readRegister(uint8_t reg) {
  I2C_0.beginTransmission(MPU6050_ADDR);
  I2C_0.write(reg);
  I2C_0.endTransmission(false);    
  I2C_0.requestFrom(MPU6050_ADDR, 1, true);
  return I2C_0.read();
}

void setup() {
  Serial.begin(115200);
  I2C_0.begin(21, 22, 400000); // Explicitly set SDA, SCL, speed
  delay(100);

  uint8_t deviceId = readRegister(WHO_AM_I);
  Serial.print("Device ID: 0x");
  Serial.println(deviceId, HEX);

  writeRegister(PWR_MGMT_1, 0x00); // Wake up
  writeRegister(0x1D, 0x03);       // Set accelerometer DLPF to 44.8Hz

  Serial.println("MPU6050 initialized!");
}

void loop() {
  const uint8_t numSamples = 25;
  int32_t avgX = 0, avgY = 0, avgZ = 0;

  for (int i = 0; i < numSamples; i++) {
    readAccelData();
    avgX += accelX;
    avgY += accelY;
    avgZ += accelZ;
    delay(1); // Short delay between samples
  }

  accelX = avgX / numSamples;
  accelY = avgY / numSamples;
  accelZ = avgZ / numSamples;
  
  // Print raw data
  Serial.print("Accel X: "); Serial.print(accelX);
  Serial.print(" | Y: ");    Serial.print(accelY);
  Serial.print(" | Z: ");    Serial.print(accelZ);
  Serial.println();

  delay(20); // Adjust delay as needed
}
