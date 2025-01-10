#ifndef BMP180_H
#define BMP180_H

#include <Arduino.h>
#include <Wire.h>
#include <I2Cdev.h>

#define BMP180_ADDRESS 0x77
#define BMP180_CONTROL_REGISTER 0xF4
#define BMP180_DATA_REGISTER 0xF6
#define BMP180_TEMP_COMMAND 0x2E
#define BMP180_PRESSURE_COMMAND 0x34

using namespace I2Cdev;

class BMP180 {
  public:
    enum class Bmp180State {
      IDLE,
      REQUESTED_TEMP,
      REQUESTED_PRESSURE
    };

    BMP180(TwoWire &w);
    int begin(int oversampling = 3);

    boolean hasNewData(){ return newData; };
    float getTemperature(){ newData = false; return temperature; };
    float getPressure(){ newData = false; return pressure; };
    float getAltitude(){ newData = false; return altitude; };
    Bmp180State getState(){ return bmp180State; };

    void fetchData ();
    void forceFetchData();

    void stateIdle();
    void stateTemperature();
    void statePressure();

  private:
    uint8_t oversampling;
    TwoWire& wire;
    boolean newData;

    Bmp180State bmp180State;
    unsigned long bmp180RequestTime;
    int16_t ac1, ac2, ac3, b1, b2, mb, mc, md;
    uint16_t ac4, ac5, ac6;
    int32_t b5;

    int32_t pressure;
    float temperature, altitude;
};

#endif