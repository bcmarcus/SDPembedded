#include "MPU6050.h"
#include "Arduino.h"

/* INIT and BASIC FUNCTIONS */

MPU6050::MPU6050(TwoWire &w) : wire(w){
  setFilterGyroCoef(DEFAULT_GYRO_COEFF);
  setGyroOffsets(0,0,0);
  setAccOffsets(0,0,0);
	
	beta = DEFAULT_BETA;
	q0 = 1.0f;
	q1 = 0.0f;
	q2 = 0.0f;
	q3 = 0.0f;
}

byte MPU6050::begin(int gyro_config_num, int acc_config_num){
  this->wire.begin();
  this->wire.setClock(32000000);
	lastCall = micros();
	deltaTime = 0;

  writeData(wire, MPU6050_ADDR, MPU6050_SMPLRT_DIV_REGISTER, 0x00);
  writeData(wire, MPU6050_ADDR, MPU6050_CONFIG_REGISTER, 0x00);

	// Write a 0x80 to the power management register to reset the sensor
	writeData(wire, MPU6050_ADDR, MPU6050_PWR_MGMT_1_REGISTER, 0x80);
	delay(100); // Give it some time to reset

	// Write a 0x00 to the power management register to wake the sensor up
	writeData(wire, MPU6050_ADDR, MPU6050_PWR_MGMT_1_REGISTER, 0x00);
	delay(100); // Give it some time to wake up

  setGyroConfig(gyro_config_num);
  setAccConfig(acc_config_num);
  byte status = writeData(wire, MPU6050_ADDR, MPU6050_PWR_MGMT_1_REGISTER, 0x01); // check only the last connection with status
  
	this->calibrate(true, true);

  this->updateFast();
  angleX = this->getAccAngleX();
  angleY = this->getAccAngleY();
  angleZ = 0;
  preInterval = millis(); // may cause issue if begin() is much before the first update()

  return status;
}

/* SETTER */

byte MPU6050::setGyroConfig(int config_num){
  byte status;
  switch(config_num){
    case 0: // range = +- 250 °/s
	  gyro_lsb_to_degsec = 131.0;
	  status = writeData(wire, MPU6050_ADDR, MPU6050_GYRO_CONFIG_REGISTER, 0x00);
	  break;
	case 1: // range = +- 500 °/s
	  gyro_lsb_to_degsec = 65.5;
	  status = writeData(wire, MPU6050_ADDR, MPU6050_GYRO_CONFIG_REGISTER, 0x08);
	  break;
	case 2: // range = +- 1000 °/s
	  gyro_lsb_to_degsec = 32.8;
	  status = writeData(wire, MPU6050_ADDR, MPU6050_GYRO_CONFIG_REGISTER, 0x10);
	  break;
	case 3: // range = +- 2000 °/s
	  gyro_lsb_to_degsec = 16.4;
	  status = writeData(wire, MPU6050_ADDR, MPU6050_GYRO_CONFIG_REGISTER, 0x18);
	  break;
	default: // error
	  status = 1;
	  break;
  }
  return status;
}

byte MPU6050::setAccConfig(int config_num){
  byte status;
  switch(config_num){
    case 0: // range = +- 2 g
	  acc_lsb_to_g = 16384.0;
	  status = writeData(wire, MPU6050_ADDR, MPU6050_ACCEL_CONFIG_REGISTER, 0x00);
	  break;
	case 1: // range = +- 4 g
	  acc_lsb_to_g = 8192.0;
	  status = writeData(wire, MPU6050_ADDR, MPU6050_ACCEL_CONFIG_REGISTER, 0x08);
	  break;
	case 2: // range = +- 8 g
	  acc_lsb_to_g = 4096.0;
	  status = writeData(wire, MPU6050_ADDR, MPU6050_ACCEL_CONFIG_REGISTER, 0x10);
	  break;
	case 3: // range = +- 16 g
	  acc_lsb_to_g = 2048.0;
	  status = writeData(wire, MPU6050_ADDR, MPU6050_ACCEL_CONFIG_REGISTER, 0x18);
	  break;
	default: // error
	  status = 1;
	  break;
  }
  return status;
}

void MPU6050::setGyroOffsets(float x, float y, float z){
  gyroXoffset = x;
  gyroYoffset = y;
  gyroZoffset = z;
}

void MPU6050::setAccOffsets(float x, float y, float z){
  accXoffset = x;
  accYoffset = y;
  accZoffset = z;
}

void MPU6050::setFilterGyroCoef(float gyro_coeff){
  if ((gyro_coeff<0) or (gyro_coeff>1)){ gyro_coeff = DEFAULT_GYRO_COEFF; } // prevent bad gyro coeff, should throw an error...
  filterGyroCoef = gyro_coeff;
}

void MPU6050::setFilterAccCoef(float acc_coeff){
  setFilterGyroCoef(1.0-acc_coeff);
}

void MPU6050::resetMagwick() {
	q0 = 1;
	q1 = 0;
	q2 = 0;
	q3 = 0;

	roll = 0;
	pitch = 0;
	yaw = 0;

	absolute_yaw = 0;
	previous_yaw = 0;
}

void MPU6050::resetFast() {
	resetAngleX();
	resetAngleY();
	resetAngleZ();
	positionX = 0;
	positionY = 0;
	positionZ = 0;
	velocityX = 0;
	velocityY = 0;
	velocityZ = 0;
}

void MPU6050::resetAll() {
	resetMagwick();
	resetFast();
}

/* CALC OFFSET */

void MPU6050::calibrate(bool is_calc_gyro, bool is_calc_acc){
  if(is_calc_gyro){ setGyroOffsets(0,0,0); }
  if(is_calc_acc){ setAccOffsets(0,0,0); }
  float ag[6] = {0,0,0,0,0,0}; // 3*acc, 3*gyro
  
  for(int i = 0; i < CALIB_OFFSET_NB_MES; i++){
    this->fetchData();
    ag[0] += accX;
    ag[1] += accY;
    ag[2] += (accZ-1.0);
    ag[3] += gyroX;
    ag[4] += gyroY;
    ag[5] += gyroZ;
    delay(1); // wait a little bit between 2 measurements
  }
  
  if(is_calc_acc){
    accXoffset = ag[0] / CALIB_OFFSET_NB_MES;
    accYoffset = ag[1] / CALIB_OFFSET_NB_MES;
    accZoffset = ag[2] / CALIB_OFFSET_NB_MES;
  }
  
  if(is_calc_gyro){
    gyroXoffset = ag[3] / CALIB_OFFSET_NB_MES;
    gyroYoffset = ag[4] / CALIB_OFFSET_NB_MES;
    gyroZoffset = ag[5] / CALIB_OFFSET_NB_MES;
  }
}

/* UPDATE */

void MPU6050::fetchData(){
	deltaTime = (micros() - lastCall) / 1000000.0; // convert to seconds
	lastCall = micros();

  wire.beginTransmission(MPU6050_ADDR);
  wire.write(MPU6050_ACCEL_OUT_REGISTER);
  wire.endTransmission(false);
  wire.requestFrom((int)MPU6050_ADDR, 14);

  int16_t rawData[7]; // [ax,ay,az,temp,gx,gy,gz]

  for(int i=0;i<7;i++){
	  rawData[i]  = wire.read() << 8;
    rawData[i] |= wire.read();
  }

  accX = ((float)rawData[0]) / acc_lsb_to_g - accXoffset;
  accY = ((float)rawData[1]) / acc_lsb_to_g - accYoffset;
  accZ = ((float)rawData[2]) / acc_lsb_to_g - accZoffset;
  temp = (rawData[3] + TEMP_LSB_OFFSET) / TEMP_LSB_2_DEGREE;
  gyroX = ((float)rawData[4]) / gyro_lsb_to_degsec - gyroXoffset;
  gyroY = ((float)rawData[5]) / gyro_lsb_to_degsec - gyroYoffset;
  gyroZ = ((float)rawData[6]) / gyro_lsb_to_degsec - gyroZoffset;

	computePosition();
}

void MPU6050::updateFast(){
  // retrieve raw data
  this->fetchData();
  
  // process data to get angles
  float sgZ = (accZ>=0)-(accZ<0);
  angleAccX = atan2(accY, sgZ*sqrt(accZ*accZ + accX*accX)) * RAD_2_DEG;
  angleAccY = - atan2(accX, sqrt(accZ*accZ + accY*accY)) * RAD_2_DEG;

  unsigned long Tnew = millis();
  float dt = (Tnew - preInterval) * 1e-3;
  preInterval = Tnew;

  angleX = (filterGyroCoef*(angleX + gyroX*dt)) + ((1.0-filterGyroCoef)*angleAccX);
  angleY = (filterGyroCoef*(angleY + gyroY*dt)) + ((1.0-filterGyroCoef)*angleAccY);
  angleZ += gyroZ*dt;
}

void MPU6050::update(float mx, float my, float mz) {
	float recipNorm;
	float s0, s1, s2, s3;
	float qDot1, qDot2, qDot3, qDot4;
	float hx, hy;
	float _2q0mx, _2q0my, _2q0mz, _2q1mx, _2bx, _2bz, _4bx, _4bz, _2q0, _2q1, _2q2, _2q3, _2q0q2, _2q2q3, q0q0, q0q1, q0q2, q0q3, q1q1, q1q2, q1q3, q2q2, q2q3, q3q3;
  float gx, gy, gz, ax, ay, az;

	// Serial.println(deltaTime);
	// Use IMU algorithm if magnetometer measurement invalid (avoids NaN in magnetometer normalisation)

	if((mx == 0.0f) && (my == 0.0f) && (mz == 0.0f)) {
		updateIMU();
		return;
	}

	this->updateFast();
	invSampleFreq = deltaTime;

	if (invSampleFreq > 0.5) {
		return;
	}

	ax = this->getAccX();
  ay = this->getAccY();
  az = this->getAccZ();
  gx = this->getGyroX();
  gy = this->getGyroY();
  gz = this->getGyroZ();

	// Convert gyroscope degrees/sec to radians/sec
	gx *= 0.0174533f;
	gy *= 0.0174533f;
	gz *= 0.0174533f;

	// Rate of change of quaternion from gyroscope
	qDot1 = 0.5f * (-q1 * gx - q2 * gy - q3 * gz);
	qDot2 = 0.5f * (q0 * gx + q2 * gz - q3 * gy);
	qDot3 = 0.5f * (q0 * gy - q1 * gz + q3 * gx);
	qDot4 = 0.5f * (q0 * gz + q1 * gy - q2 * gx);

	// Compute feedback only if accelerometer measurement valid (avoids NaN in accelerometer normalisation)
	if(!((ax == 0.0f) && (ay == 0.0f) && (az == 0.0f))) {

		// Normalise accelerometer measurement
		recipNorm = invSqrt(ax * ax + ay * ay + az * az);
		ax *= recipNorm;
		ay *= recipNorm;
		az *= recipNorm;

		// Normalise magnetometer measurement
		recipNorm = invSqrt(mx * mx + my * my + mz * mz);
		mx *= recipNorm;
		my *= recipNorm;
		mz *= recipNorm;

		// Auxiliary variables to avoid repeated arithmetic
		_2q0mx = 2.0f * q0 * mx;
		_2q0my = 2.0f * q0 * my;
		_2q0mz = 2.0f * q0 * mz;
		_2q1mx = 2.0f * q1 * mx;
		_2q0 = 2.0f * q0;
		_2q1 = 2.0f * q1;
		_2q2 = 2.0f * q2;
		_2q3 = 2.0f * q3;
		_2q0q2 = 2.0f * q0 * q2;
		_2q2q3 = 2.0f * q2 * q3;
		q0q0 = q0 * q0;
		q0q1 = q0 * q1;
		q0q2 = q0 * q2;
		q0q3 = q0 * q3;
		q1q1 = q1 * q1;
		q1q2 = q1 * q2;
		q1q3 = q1 * q3;
		q2q2 = q2 * q2;
		q2q3 = q2 * q3;
		q3q3 = q3 * q3;

		// Reference direction of Earth's magnetic field
		hx = mx * q0q0 - _2q0my * q3 + _2q0mz * q2 + mx * q1q1 + _2q1 * my * q2 + _2q1 * mz * q3 - mx * q2q2 - mx * q3q3;
		hy = _2q0mx * q3 + my * q0q0 - _2q0mz * q1 + _2q1mx * q2 - my * q1q1 + my * q2q2 + _2q2 * mz * q3 - my * q3q3;
		_2bx = sqrtf(hx * hx + hy * hy);
		_2bz = -_2q0mx * q2 + _2q0my * q1 + mz * q0q0 + _2q1mx * q3 - mz * q1q1 + _2q2 * my * q3 - mz * q2q2 + mz * q3q3;
		_4bx = 2.0f * _2bx;
		_4bz = 2.0f * _2bz;

		// Gradient decent algorithm corrective step
		s0 = -_2q2 * (2.0f * q1q3 - _2q0q2 - ax) + _2q1 * (2.0f * q0q1 + _2q2q3 - ay) - _2bz * q2 * (_2bx * (0.5f - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx) + (-_2bx * q3 + _2bz * q1) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my) + _2bx * q2 * (_2bx * (q0q2 + q1q3) + _2bz * (0.5f - q1q1 - q2q2) - mz);
		s1 = _2q3 * (2.0f * q1q3 - _2q0q2 - ax) + _2q0 * (2.0f * q0q1 + _2q2q3 - ay) - 4.0f * q1 * (1 - 2.0f * q1q1 - 2.0f * q2q2 - az) + _2bz * q3 * (_2bx * (0.5f - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx) + (_2bx * q2 + _2bz * q0) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my) + (_2bx * q3 - _4bz * q1) * (_2bx * (q0q2 + q1q3) + _2bz * (0.5f - q1q1 - q2q2) - mz);
		s2 = -_2q0 * (2.0f * q1q3 - _2q0q2 - ax) + _2q3 * (2.0f * q0q1 + _2q2q3 - ay) - 4.0f * q2 * (1 - 2.0f * q1q1 - 2.0f * q2q2 - az) + (-_4bx * q2 - _2bz * q0) * (_2bx * (0.5f - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx) + (_2bx * q1 + _2bz * q3) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my) + (_2bx * q0 - _4bz * q2) * (_2bx * (q0q2 + q1q3) + _2bz * (0.5f - q1q1 - q2q2) - mz);
		s3 = _2q1 * (2.0f * q1q3 - _2q0q2 - ax) + _2q2 * (2.0f * q0q1 + _2q2q3 - ay) + (-_4bx * q3 + _2bz * q1) * (_2bx * (0.5f - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx) + (-_2bx * q0 + _2bz * q2) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my) + _2bx * q1 * (_2bx * (q0q2 + q1q3) + _2bz * (0.5f - q1q1 - q2q2) - mz);
		recipNorm = invSqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3); // normalise step magnitude
		s0 *= recipNorm;
		s1 *= recipNorm;
		s2 *= recipNorm;
		s3 *= recipNorm;

		// Apply feedback step
		qDot1 -= beta * s0;
		qDot2 -= beta * s1;
		qDot3 -= beta * s2;
		qDot4 -= beta * s3;
	}

	// Integrate rate of change of quaternion to yield quaternion
	q0 += qDot1 * invSampleFreq;
	q1 += qDot2 * invSampleFreq;
	q2 += qDot3 * invSampleFreq;
	q3 += qDot4 * invSampleFreq;

	// Normalise quaternion
	recipNorm = invSqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
	q0 *= recipNorm;
	q1 *= recipNorm;
	q2 *= recipNorm;
	q3 *= recipNorm;
	anglesComputed = 0;
}


void MPU6050::updateIMU() {
	float recipNorm;
	float s0, s1, s2, s3;
	float qDot1, qDot2, qDot3, qDot4;
	float _2q0, _2q1, _2q2, _2q3, _4q0, _4q1, _4q2 ,_8q1, _8q2, q0q0, q1q1, q2q2, q3q3;
  float gx, gy, gz, ax, ay, az; 

	this->updateFast();

	ax = this->getAccX();
  ay = this->getAccY();
  az = this->getAccZ();
  gx = this->getGyroX();
  gy = this->getGyroY();
  gz = this->getGyroZ();

  invSampleFreq = deltaTime;

	if (invSampleFreq > 0.5) {
		return;
	}

	// Convert gyroscope degrees/sec to radians/sec
	gx *= 0.0174533f;
	gy *= 0.0174533f;
	gz *= 0.0174533f;

	// Rate of change of quaternion from gyroscope
	qDot1 = 0.5f * (-q1 * gx - q2 * gy - q3 * gz);
	qDot2 = 0.5f * (q0 * gx + q2 * gz - q3 * gy);
	qDot3 = 0.5f * (q0 * gy - q1 * gz + q3 * gx);
	qDot4 = 0.5f * (q0 * gz + q1 * gy - q2 * gx);

	// Compute feedback only if accelerometer measurement valid (avoids NaN in accelerometer normalisation)
	if(!((ax == 0.0f) && (ay == 0.0f) && (az == 0.0f))) {

		// Normalise accelerometer measurement
		recipNorm = invSqrt(ax * ax + ay * ay + az * az);
		ax *= recipNorm;
		ay *= recipNorm;
		az *= recipNorm;

		// Auxiliary variables to avoid repeated arithmetic
		_2q0 = 2.0f * q0;
		_2q1 = 2.0f * q1;
		_2q2 = 2.0f * q2;
		_2q3 = 2.0f * q3;
		_4q0 = 4.0f * q0;
		_4q1 = 4.0f * q1;
		_4q2 = 4.0f * q2;
		_8q1 = 8.0f * q1;
		_8q2 = 8.0f * q2;
		q0q0 = q0 * q0;
		q1q1 = q1 * q1;
		q2q2 = q2 * q2;
		q3q3 = q3 * q3;

		// Gradient decent algorithm corrective step
		s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay;
		s1 = _4q1 * q3q3 - _2q3 * ax + 4.0f * q0q0 * q1 - _2q0 * ay - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az;
		s2 = 4.0f * q0q0 * q2 + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az;
		s3 = 4.0f * q1q1 * q3 - _2q1 * ax + 4.0f * q2q2 * q3 - _2q2 * ay;
		recipNorm = invSqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3); // normalise step magnitude
		s0 *= recipNorm;
		s1 *= recipNorm;
		s2 *= recipNorm;
		s3 *= recipNorm;

		// Apply feedback step
		qDot1 -= beta * s0;
		qDot2 -= beta * s1;
		qDot3 -= beta * s2;
		qDot4 -= beta * s3;
	}

	// Integrate rate of change of quaternion to yield quaternion
	q0 += qDot1 * invSampleFreq;
	q1 += qDot2 * invSampleFreq;
	q2 += qDot3 * invSampleFreq;
	q3 += qDot4 * invSampleFreq;

	// Normalise quaternion
	recipNorm = invSqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
	q0 *= recipNorm;
	q1 *= recipNorm;
	q2 *= recipNorm;
	q3 *= recipNorm;
	anglesComputed = 0;
}


//-------------------------------------------------------------------------------------------
// Fast inverse square-root
// See: http://en.wikipedia.org/wiki/Fast_inverse_square_root

// float MPU6050::invSqrt(float x) {
// 	float halfx = 0.5f * x;
// 	float y = x;
// 	long i = *(long*)&y;
// 	i = 0x5f3759df - (i>>1);
// 	y = *(float*)&i;
// 	y = y * (1.5f - (halfx * y * y));
// 	y = y * (1.5f - (halfx * y * y));
// 	return y;
// }

float MPU6050::invSqrt(float x) {
	return 1 / sqrt(x);
}

//-------------------------------------------------------------------------------------------

void MPU6050::computeAngles()
{
	roll = atan2f(q0*q1 + q2*q3, 0.5f - q1*q1 - q2*q2);
	pitch = asinf(-2.0f * (q1*q3 - q0*q2));
	float new_yaw = atan2f(q1*q2 + q0*q3, 0.5f - q2*q2 - q3*q3);

	// Calculate yaw difference
	float yaw_diff = new_yaw - previous_yaw;
	// Detect and handle the wrap-around case
	if (yaw_diff > M_PI) yaw_diff -= 2 * M_PI;
	if (yaw_diff < -M_PI) yaw_diff += 2 * M_PI;

	// Convert to degrees and add to absolute yaw
	absolute_yaw += yaw_diff;
	previous_yaw = new_yaw;

	yaw = new_yaw;
	anglesComputed = 1;
}

void MPU6050::computePosition() {
  // Compute the acceleration components in the Earth frame
	double ax, ay, az;
	ax = accX * 9.81;
	ay = accY * 9.81;
	az = (accZ - 1) * 9.81;

	double p, r;
	p = getPitchRadians();
	r = getRollRadians();

  double earthAccX = ax * cos(p) + ay * sin(r) * sin(p) + az * cos(r) * sin(p);
  double earthAccY = ay * cos(r) - az * sin(r);
  double earthAccZ = az * (-sin(p)) + ay * sin(r) * cos(p) + az * cos(r) * cos(p);

  // Update velocity and position
  velocityX += earthAccX * deltaTime;
  velocityY += earthAccY * deltaTime;
  velocityZ += earthAccZ * deltaTime;

  positionX += velocityX * deltaTime;
  positionY += velocityY * deltaTime;
  positionZ += velocityZ * deltaTime;
}