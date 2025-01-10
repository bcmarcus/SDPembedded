#include "GY87.h"
#include "Arduino.h"

/* INIT and BASIC FUNCTIONS */

GY87::GY87(TwoWire &w) : compass(w), bmp(w), mpu(w) {
  wire = &w;
  setFilterGyroCoef(DEFAULT_GYRO_COEFF);
  setGyroOffsets(0,0,0);
  setAccOffsets(0,0,0);
}

// void GY87::setI2CBypass(bool enable) {
//   uint8_t intPinCfg;

//   // Read the current value of the INT_PIN_CFG register
//   wire->beginTransmission(MPU6050_ADDR);
//   wire->write(MPU6050_REG_INT_PIN_CFG);
//   wire->endTransmission(false); // Send a repeated start
//   wire->requestFrom(MPU6050_ADDR, 1);
//   intPinCfg = wire->read();
//   wire->endTransmission();

//   // Modify the I2C_BYPASS_EN bit
//   if (enable) {
//     intPinCfg |= MPU6050_BIT_I2C_BYPASS_EN;
//   } else {
//     intPinCfg &= ~MPU6050_BIT_I2C_BYPASS_EN;
//   }

//   // Write the modified value back to the INT_PIN_CFG register
//   wire->beginTransmission(MPU6050_ADDR);
//   wire->write(MPU6050_REG_INT_PIN_CFG);
//   wire->write(intPinCfg);
//   wire->endTransmission();
// }

byte GY87::begin(int gyro_config_num, int acc_config_num, int bmp_resolution, boolean setI2CBypass){
  wire->begin();

	// begin mpu
  if (mpu.begin(gyro_config_num, acc_config_num) != 0) {
		// Serial.println("Mpu failed to start");
		return 1;
	}

  // begin bmp
  if (bmp.begin(bmp_resolution) != 0) {
		// Serial.println("Bmp failed to start");
		return 1;
	}

  // begin mag
  if (compass.begin(setI2CBypass) != 0) {
		// Serial.println("Compass failed to start");
		return 1;
	}

  // calibrate everything
  this->calibrate();

	wire->setClock(8000000);

  this->updateFast();
  mpu.resetAngleX();
	mpu.resetAngleY();
	mpu.resetAngleZ();
  return 0;
}

/* CALC OFFSET */

boolean GY87::calibrate(boolean calibrateMpu, boolean calibrateCompass){
	mpu.resetAngleX();
	mpu.resetAngleY();
	mpu.resetAngleZ();

	if (calibrateMpu){
		mpu.calibrate(true, true);
		stabilize(true);
	}
	if (calibrateCompass) {
		compass.calibrate();
		stabilize(false);
	}

	mpu.resetAngleX();
	mpu.resetAngleY();
	mpu.resetAngleZ();

	return true;
}

void GY87::stabilize(bool imuOnly, long time) {
	long long startTime = millis();
	
	this->resetAll();

	float prevBeta = this->getBeta();

	this->setBeta(0.8);

	while (millis() - startTime < time){
		if (imuOnly) {
			this->updateIMU();

			this->getRoll();
			this->getPitch();
			this->getAbsoluteYaw();
		} else {
			this->update();
			
			this->getRoll();
			this->getPitch();
			this->getAbsoluteYaw();
		}
	}

	this->setBeta(prevBeta);
}

/* UPDATE */

void GY87::fetchData(){
  mpu.fetchData();
	compass.fetchData();
	bmp.fetchData();
}

void GY87::updateFast(){
  // retrieve raw data
  this->fetchData();
	mpu.updateFast();
}