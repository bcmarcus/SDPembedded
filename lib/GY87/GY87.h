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

#ifndef GY87_H
#define GY87_H

#include "Arduino.h"
#include "Wire.h"
#include "I2Cdev.h"
#include "HMC5883L.h"
#include "BMP180.h"
#include "MPU6050.h"

#define MPU6050_ADDR                  0x68
#define MPU6050_SMPLRT_DIV_REGISTER   0x19
#define MPU6050_CONFIG_REGISTER       0x1a
#define MPU6050_GYRO_CONFIG_REGISTER  0x1b
#define MPU6050_ACCEL_CONFIG_REGISTER 0x1c
#define MPU6050_PWR_MGMT_1_REGISTER   0x6b
#define MPU6050_REG_INT_PIN_CFG 0x37
#define MPU6050_BIT_I2C_BYPASS_EN 0x02

#define MPU6050_GYRO_OUT_REGISTER     0x43
#define MPU6050_ACCEL_OUT_REGISTER    0x3B

#define RAD_2_DEG             57.29578 // [°/rad]
#define CALIBRATION_TIME      2500
#define TEMP_LSB_2_DEGREE     340.0    // [bit/celsius]
#define TEMP_LSB_OFFSET       12412.0

#define DEFAULT_GYRO_COEFF    0.98

class GY87{
  public:
    // INIT and BASIC FUNCTIONS
    GY87(TwoWire &w);
    byte begin(int gyro_config_num=1, int acc_config_num=0, int bmp_resolution = 3, boolean setI2CBypass = true);

    boolean calibrate(boolean calibrateMpu = true, boolean calibrateCompass = true);
    void stabilize (bool imuOnly, long time = CALIBRATION_TIME);

    void resetMagwick() {mpu.resetMagwick();};
    void resetFast(){mpu.resetFast();};
    void resetAll(){mpu.resetAll();};

    void calcOffsets(bool is_calc_gyro=true, bool is_calc_acc=true);
    
    // MPU CONFIG SETTER
    void setBeta(float beta) {mpu.setBeta(beta);};
    float getBeta() {return mpu.getBeta();};
    
    void setMagOffsets(float hardIronX, float hardIronY, float hardIronZ, float softIronX, float softIronY, float softIronZ);
    // void setBmpOffsets(float temp, float pres, float alt);

    /* SETTER */
    byte setGyroConfig(int config_num){return mpu.setGyroConfig(config_num);};
    byte setAccConfig(int config_num){return mpu.setAccConfig(config_num);};
    void setGyroOffsets(float x, float y, float z){mpu.setGyroOffsets(x, y, z);};
    void setAccOffsets(float x, float y, float z){mpu.setAccOffsets(x, y, z);};
    void setFilterGyroCoef(float gyro_coeff){mpu.setFilterGyroCoef(gyro_coeff);};
    void setFilterAccCoef(float acc_coeff){mpu.setFilterAccCoef(acc_coeff);};

    // MPU CONFIG GETTER
    float getGyroXoffset(){ return mpu.getGyroXoffset(); };
    float getGyroYoffset(){ return mpu.getGyroYoffset(); };
    float getGyroZoffset(){ return mpu.getGyroZoffset(); };
    
    float getAccXoffset(){ return mpu.getAccXoffset(); };
    float getAccYoffset(){ return mpu.getAccYoffset(); };
    float getAccZoffset(){ return mpu.getAccZoffset(); };
    
    float getFilterGyroCoef(){ return mpu.getFilterGyroCoef(); };
    float getFilterAccCoef(){ return mpu.getFilterAccCoef(); };

    void setSoftIronScale(float x, float y, float z) {this->compass.setSoftIronScale(x, y, z);};
    void getSoftIronScale(float &x, float &y, float &z) {this->compass.getSoftIronScale(x, y, z);};
    void setHardIronBias(float x, float y, float z) {this->compass.setHardIronBias(x, y, z);};
    void getHardIronBias(float &x, float &y, float &z) {this->compass.getHardIronBias(x, y, z);};
    
    // INLOOP GETTER
    float getPositionX(){ return mpu.getPositionX(); };
    float getPositionY(){ return mpu.getPositionY(); };
    float getPositionZ(){ return mpu.getPositionZ(); };

    float getVelocityX(){ return mpu.getVelocityX(); };
    float getVelocityY(){ return mpu.getVelocityY(); };
    float getVelocityZ(){ return mpu.getVelocityZ(); };

    float getAccX(){ return mpu.getAccX(); };
    float getAccY(){ return mpu.getAccY(); };
    float getAccZ(){ return mpu.getAccZ(); };

    float getGyroX(){ return mpu.getGyroX(); };
    float getGyroY(){ return mpu.getGyroY(); };
    float getGyroZ(){ return mpu.getGyroZ(); };

    float getMagX(){ return compass.getX(); };
    float getMagY(){ return compass.getY(); };
    float getMagZ(){ return compass.getZ(); };

    float getTemp(){ return bmp.getTemperature(); };
    float getPressure(){ return bmp.getPressure(); };
    float getAltitude(){ return bmp.getAltitude(); };

    float getAccAngleX(){ return mpu.getAccAngleX(); };
    float getAccAngleY(){ return mpu.getAccAngleY(); };

    float getAngleX(){ return mpu.getAngleX(); };
    float getAngleY(){ return mpu.getAngleY(); };
    float getAngleZ(){ return mpu.getAngleZ(); };

    void resetAngleX() { updateFast (); mpu.resetAngleX(); };
    void resetAngleY() { updateFast (); mpu.getAccAngleY(); };
    void resetAngleZ() { mpu.resetAngleZ(); };
    // INLOOP UPDATE
    void fetchData(); // user should better call 'update' that includes 'fetchData'
    void updateFast();
    void update() {mpu.update(getMagX(), getMagY(), getMagZ());};
    void updateIMU() {mpu.updateIMU();};
    
    float getRoll() {return mpu.getRoll();};
    float getPitch() {return mpu.getPitch();};
    float getYaw() {return mpu.getYaw();};
    float getAbsoluteYaw() {return mpu.getAbsoluteYaw();};

    float getRollRadians() {return mpu.getRollRadians();};
    float getPitchRadians() {return mpu.getPitchRadians();};
    float getYawRadians() {return mpu.getYawRadians();};

  private:
    TwoWire *wire;
    HMC5883L compass;
    BMP180 bmp;
    MPU6050 mpu;
};

#endif
