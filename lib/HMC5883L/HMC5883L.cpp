#include "HMC5883L.h"
#include <Arduino.h>

HMC5883L::HMC5883L(TwoWire &wire) : wire(wire) {
  this->hardIronBiasX = 0;
  this->hardIronBiasY = 0;
  this->hardIronBiasZ = 0;
  this->softIronScaleX = 1;
  this->softIronScaleY = 1;
  this->softIronScaleZ = 1;
}

void HMC5883L::setSoftIronScale(float x, float y, float z) {
    softIronScaleX = x;
    softIronScaleY = y;
    softIronScaleZ = z;
}

void HMC5883L::getSoftIronScale(float &x, float &y, float &z) {
    x = softIronScaleX;
    y = softIronScaleY;
    z = softIronScaleZ;
}

void HMC5883L::setHardIronBias(float x, float y, float z) {
    hardIronBiasX = x;
    hardIronBiasY = y;
    hardIronBiasZ = z;
}

void HMC5883L::getHardIronBias(float &x, float &y, float &z) {
    x = hardIronBiasX;
    y = hardIronBiasY;
    z = hardIronBiasZ;
}

int HMC5883L::begin(boolean bypass) {
  wire.begin();

  if (bypass) {
    // Wake up MPU6050
    writeData(wire, MPU6050_ADDRESS, 0x6B, 0x00);

    // Enable I2C bypass on MPU6050
    writeData(wire, MPU6050_ADDRESS, 0x37, 0x02);
  }
  this->bypass = bypass;

  wire.setClock(6800000);
  return 0;
}
void HMC5883L::selectionSort(float* data, int length) {
  int i, j, minIndex;
  float tmp; 
  for (i = 0; i < length - 1; i++) {
    minIndex = i;
    for (j = i + 1; j < length; j++)
      if (data[j] < data[minIndex])
        minIndex = j;
    if (minIndex != i) {
      tmp = data[i];
      data[i] = data[minIndex];
      data[minIndex] = tmp;
    }
  }
}

float HMC5883L::TukeysFences(float* data, int length) {
  // First sort the data
  selectionSort(data, length);

  // Calculate first quartile (Q1) and third quartile (Q3)
  int q1Index = length / 4;
  int q3Index = 3 * length / 4;
  float q1 = data[q1Index];
  float q3 = data[q3Index];

  // Calculate interquartile range (IQR)
  float iqr = q3 - q1;

  // Calculate lower fence (Q1 - 1.5 * IQR)
  float lowerFence = q1 - 1.5 * iqr;

  return lowerFence;
}

float HMC5883L::TukeysFencesMax(float* data, int length) {
  // First sort the data
  selectionSort(data, length);

  // Calculate first quartile (Q1) and third quartile (Q3)
  int q1Index = length / 4;
  int q3Index = 3 * length / 4;
  float q1 = data[q1Index];
  float q3 = data[q3Index];

  // Calculate interquartile range (IQR)
  float iqr = q3 - q1;

  // Calculate upper fence (Q3 + 1.5 * IQR)
  float upperFence = q3 + 1.5 * iqr;

  return upperFence;
}


float HMC5883L::KalmanFilter(float input, float* prev, float* p, float q, float r) {
  // Predict
  *p += q;

  // Update
  float k = *p / (*p + r);
  float x = *prev + k * (input - *prev);
  *p = (1 - k) * (*p);

  *prev = x;
  return x;
}

void HMC5883L::fetchData() {
  // Trigger a single measurement in HMC5883L
  writeData(wire, HMC5883L_ADDRESS, 0x02, 0x01);

  // Wait for Data Ready bit to be set in status register
  while ((readByte(wire, HMC5883L_ADDRESS, HMC5883L_STATUS_REGISTER) & 0x01) == 0) {};

  // Read HMC5883L
  uint16_t data[3];
  readWords(wire, HMC5883L_ADDRESS, 0x03, data, 3);
  x = data[0];
  y = data[1];
  z = data[2];

  // Apply Kalman Filter
  static float prevX = 0, prevY = 0, prevZ = 0;
  static float pX = 0.1, pY = 0.1, pZ = 0.1;  // Initial estimation error covariance
  float q = 0.1;  // Process variance
  float r = 0.1;  // Measurement variance

  x = KalmanFilter(x, &prevX, &pX, q, r) - hardIronBiasX;
  y = KalmanFilter(y, &prevY, &pY, q, r) - hardIronBiasY;
  z = KalmanFilter(z, &prevZ, &pZ, q, r) - hardIronBiasZ;

  x *= softIronScaleX;
  y *= softIronScaleY;
  z *= softIronScaleZ;
}


void HMC5883L::calibrate() {
  Serial.println("Calibrating magnetometer, please move as if painting the inside of a sphere.");

  float x_vals[CALIBRATION_SAMPLES], y_vals[CALIBRATION_SAMPLES], z_vals[CALIBRATION_SAMPLES];

  // Fetch data
  for(long i = 0; i < CALIBRATION_SAMPLES; i++) {
    this->fetchData();
    x_vals[i] = x;
    y_vals[i] = y;
    z_vals[i] = z;
    delay(1); // Delay between measurements.
  }

  // Apply Tukey's Fences and compute min, max for each axis
  float minX = TukeysFences(x_vals, CALIBRATION_SAMPLES);
  float minY = TukeysFences(y_vals, CALIBRATION_SAMPLES);
  float minZ = TukeysFences(z_vals, CALIBRATION_SAMPLES);

  float maxX = TukeysFencesMax(x_vals, CALIBRATION_SAMPLES);
  float maxY = TukeysFencesMax(y_vals, CALIBRATION_SAMPLES);
  float maxZ = TukeysFencesMax(z_vals, CALIBRATION_SAMPLES);

  // Hard iron correction
  hardIronBiasX = (maxX + minX)/2; // get average x
  hardIronBiasY = (maxY + minY)/2; // get average y
  hardIronBiasZ = (maxZ + minZ)/2; // get average z

  // Soft iron correction
  float x_range = maxX - minX;
  float y_range = maxY - minY;
  float z_range = maxZ - minZ;
  
  float avg_range = (x_range + y_range + z_range) / 3;

  softIronScaleX = avg_range / x_range;
  softIronScaleY = avg_range / y_range;
  softIronScaleZ = avg_range / z_range;

  Serial.print("HardIron: X "); Serial.print(hardIronBiasX); Serial.print(" Y "); Serial.print(hardIronBiasY); Serial.print(" Z "); Serial.println(hardIronBiasZ);
  Serial.print("SoftIron: X "); Serial.print(softIronScaleX); Serial.print(" Y "); Serial.print(softIronScaleY); Serial.print(" Z "); Serial.println(softIronScaleZ);

  Serial.println("Calibration complete.");
}


void HMC5883L::calculateHeading() {
  // Assuming x, y, z are your magnetometer readings
  float pitch = atan2(-x, sqrt(y * y + z * z));
  float roll = atan2(y, z);

  // Declination angle adjustment
  float cosRoll = cos(roll);
  float sinRoll = sin(roll);
  float cosPitch = cos(pitch);
  float sinPitch = sin(pitch);

  // Tilt compensated magnetic field X
  float tiltCompX = x * cosPitch + y * sinRoll * sinPitch + z * cosRoll * sinPitch;
  // Tilt compensated magnetic field Y
  float tiltCompY = y * cosRoll - z * sinRoll;

  // Magnetic heading
  float heading = atan2(tiltCompY, tiltCompX);

  // Correct for when signs are reversed.
  if (heading < 0) heading += 2 * PI;

  // Check for wrap due to addition of declination.
  if (heading > 2 * PI) heading -= 2 * PI;

  // Convert radians to degrees for readability.
  headingDegrees = heading * 180 / M_PI;
}