// maybe implement mag gain one day

#ifndef HMC5883L_H
#define HMC5883L_H

#include <Wire.h>
#include <I2Cdev.h>

#define MPU6050_ADDRESS 0x68
#define CALIBRATION_SAMPLES 1000
#define HMC5883L_ADDRESS 0x1E
#define HMC5883L_STATUS_REGISTER 0x09

using namespace I2Cdev;

class HMC5883L {
  public:
    HMC5883L(TwoWire &wire);

    int begin(boolean bypass = true);
    void fetchData();
    void calibrate();
    void calculateHeading();

    int16_t getX() const { return x; }
    int16_t getY() const { return y; }
    int16_t getZ() const { return z; }
    float getHeading() const { return headingDegrees; }

    void setSoftIronScale(float x, float y, float z);
    void getSoftIronScale(float &x, float &y, float &z);
    void setHardIronBias(float x, float y, float z);
    void getHardIronBias(float &x, float &y, float &z);
    
    void selectionSort(float* data, int length);
    float TukeysFences(float* data, int length);
    float TukeysFencesMax(float* data, int length);
    float KalmanFilter(float input, float* prev, float* p, float q, float r);

  private:
    TwoWire &wire;
    boolean bypass;

    int16_t x, y, z;
    float headingDegrees;

    float hardIronBiasX, hardIronBiasY, hardIronBiasZ;
    float softIronScaleX, softIronScaleY, softIronScaleZ;
};

#endif // HMC5883L_H