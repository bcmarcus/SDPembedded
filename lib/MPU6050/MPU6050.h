/* The register map is provided at
 * https://invensense.tdk.com/wp-content/uploads/2015/02/MPU-6000-Register-Map1.pdf
 *
 * Mapping of the different gyro and accelero configurations:
 *
 * GYRO_CONFIG_[0,1,2,3] range = +- [250, 500,1000,2000] °/s
 *                       sensi =    [131,65.5,32.8,16.4] bit/(°/s)
 *
 * ACC_CONFIG_[0,1,2,3] range = +- [    2,   4,   8,  16] times the gravity (9.81m/s²)
 *                      sensi =    [16384,8192,4096,2048] bit/gravity
*/

#ifndef MPU6050_H
#define MPU6050_H

#include "Arduino.h"
#include "Wire.h"
#include "I2Cdev.h"

#define MPU6050_ADDR                  0x68
#define MPU6050_SMPLRT_DIV_REGISTER   0x19
#define MPU6050_CONFIG_REGISTER       0x1a
#define MPU6050_GYRO_CONFIG_REGISTER  0x1b
#define MPU6050_ACCEL_CONFIG_REGISTER 0x1c
#define MPU6050_PWR_MGMT_1_REGISTER   0x6b

#define MPU6050_GYRO_OUT_REGISTER     0x43
#define MPU6050_ACCEL_OUT_REGISTER    0x3B

#define RAD_2_DEG             57.29578 // [°/rad]
#define CALIB_OFFSET_NB_MES   2500
#define TEMP_LSB_2_DEGREE     340.0    // [bit/celsius]
#define TEMP_LSB_OFFSET       12412.0

#define DEFAULT_GYRO_COEFF    0.98
#define DEFAULT_BETA          0.046f            // 2 * proportional gain

using namespace I2Cdev;

class MPU6050{
  public:
    // INIT and BASIC FUNCTIONS
    MPU6050(TwoWire &w);
    byte begin(int gyro_config_num=1, int acc_config_num=0);
    
    void calibrate(bool is_calc_gyro=true, bool is_calc_acc=true);
    void calibrateGyroOffsets(){ calibrate(true,false); }; // retro-compatibility with v1.0.0
    void calibrateAccOffsets(){ calibrate(false,true); }; // retro-compatibility with v1.0.0
    
    // MPU CONFIG SETTER
    byte setGyroConfig(int config_num);
    byte setAccConfig(int config_num);
    
    void setGyroOffsets(float x, float y, float z);
    void setAccOffsets(float x, float y, float z);
    
    void setFilterGyroCoef(float gyro_coeff);
    void setFilterAccCoef(float acc_coeff);

    void resetMagwick();
    void resetFast();
    void resetAll();

    void setBeta(float beta) {this->beta = beta;};
    float getBeta() {return this->beta;};

    // MPU CONFIG GETTER
    float getGyroXoffset(){ return gyroXoffset; };
    float getGyroYoffset(){ return gyroYoffset; };
    float getGyroZoffset(){ return gyroZoffset; };
    
    float getAccXoffset(){ return accXoffset; };
    float getAccYoffset(){ return accYoffset; };
    float getAccZoffset(){ return accZoffset; };
    
    float getFilterGyroCoef(){ return filterGyroCoef; };
    float getFilterAccCoef(){ return 1.0-filterGyroCoef; };
    
    // INLOOP GETTER
    float getTemp(){ return temp; };

    float getPositionX(){ return positionX; };
    float getPositionY(){ return positionY; };
    float getPositionZ(){ return positionZ; };

    float getVelocityX(){ return velocityX; };
    float getVelocityY(){ return velocityY; };
    float getVelocityZ(){ return velocityZ; };

    float getAccX(){ return accX; };
    float getAccY(){ return accY; };
    float getAccZ(){ return accZ; };

    float getGyroX(){ return gyroX; };
    float getGyroY(){ return gyroY; };
    float getGyroZ(){ return gyroZ; };

    float getAccAngleX(){ return angleAccX; };
    float getAccAngleY(){ return angleAccY; };

    float getAngleX(){ return angleX; };
    float getAngleY(){ return angleY; };
    float getAngleZ(){ return angleZ; };

    float getDeltaTime(){ return deltaTime; };

    void resetAngleX() { updateFast (); angleX = getAccAngleX(); };
    void resetAngleY() { updateFast (); angleY = getAccAngleY(); };
    void resetAngleZ() { angleZ = 0; };
    
    // INLOOP UPDATE
    void fetchData(); // user should better call 'update' that includes 'fetchData'
    void updateFast();
    void update(float mx, float my, float mz);
    void updateIMU();
    
    float getRoll() {
        if (!anglesComputed) computeAngles();
        return roll * 57.29578f;
    }
    float getPitch() {
        if (!anglesComputed) computeAngles();
        return pitch * 57.29578f;
    }
    float getYaw() {
        if (!anglesComputed) computeAngles();
        return yaw * 57.29578f + 180.0f;
    }

    float getAbsoluteYaw() {
        if (!anglesComputed) computeAngles();
        return absolute_yaw * 57.29578f;
    }

    float getRollRadians() {
        if (!anglesComputed) computeAngles();
        return roll;
    }
    float getPitchRadians() {
        if (!anglesComputed) computeAngles();
        return pitch;
    }
    float getYawRadians() {
        if (!anglesComputed) computeAngles();
        return yaw;
    }

  private:
    TwoWire& wire;
	float gyro_lsb_to_degsec, acc_lsb_to_g;
    float gyroXoffset, gyroYoffset, gyroZoffset;
	float accXoffset, accYoffset, accZoffset;
    float temp, accX, accY, accZ, gyroX, gyroY, gyroZ;
    float velocityX, velocityY, velocityZ;
    float positionX, positionY, positionZ;
    float angleAccX, angleAccY;
    float angleX, angleY, angleZ;
    long preInterval;
    float filterGyroCoef; // complementary filter coefficient to balance gyro vs accelero data to get angle
    float deltaTime, lastCall;

    static float invSqrt(float x);
    float beta;				// algorithm gain
    float q0;
    float q1;
    float q2;
    float q3;	// quaternion of sensor frame relative to auxiliary frame
    float invSampleFreq;
    float roll;
    float pitch;
    float yaw;
    float previous_yaw;
    float absolute_yaw;

    char anglesComputed;
    void computeAngles();
    void computePosition();
};

#endif
