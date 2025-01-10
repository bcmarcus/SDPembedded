#include <BMP180.h>

BMP180::BMP180(TwoWire &w) 
  : wire(w), newData(false) {};

int BMP180::begin(int oversampling) {
  this->wire.begin();
  this->wire.setClock(20000000);
  this->oversampling = oversampling;

  bmp180RequestTime = millis();

  bmp180State = Bmp180State::IDLE;

  ac1 = readWord(wire, BMP180_ADDRESS, 0xAA);
  ac2 = readWord(wire, BMP180_ADDRESS, 0xAC);
  ac3 = readWord(wire, BMP180_ADDRESS, 0xAE);
  ac4 = readWord(wire, BMP180_ADDRESS, 0xB0);
  ac5 = readWord(wire, BMP180_ADDRESS, 0xB2);
  ac6 = readWord(wire, BMP180_ADDRESS, 0xB4);
  b1 = readWord(wire, BMP180_ADDRESS, 0xB6);
  b2 = readWord(wire, BMP180_ADDRESS, 0xB8);
  mb = readWord(wire, BMP180_ADDRESS, 0xBA);
  mc = readWord(wire, BMP180_ADDRESS, 0xBC);
  md = readWord(wire, BMP180_ADDRESS, 0xBE);
  
  return 0;
}

void BMP180::stateIdle() {
  // Request a temperature measurement
  writeData(wire, BMP180_ADDRESS, BMP180_CONTROL_REGISTER, BMP180_TEMP_COMMAND);
  bmp180State = Bmp180State::REQUESTED_TEMP;
  bmp180RequestTime = millis();
  bmp180State = Bmp180State::REQUESTED_TEMP;
}

void BMP180::stateTemperature() {
  if (millis() - this->bmp180RequestTime >= 5) { 
    // Read the temperature data
    int32_t temp = readWord(wire, BMP180_ADDRESS, BMP180_DATA_REGISTER);

    int32_t x1 = ((temp - ac6) * ac5) >> 15;
    int32_t x2 = (mc << 11) / (x1 + md);
    b5 = x1 + x2;
    temperature = ((b5 + 8) >> 4) / 10.0; // Temperature in Celsius

    // Request a pressure measurement
    writeData(wire, BMP180_ADDRESS, BMP180_CONTROL_REGISTER, BMP180_PRESSURE_COMMAND + (this->oversampling << 6));
    bmp180State = Bmp180State::REQUESTED_PRESSURE;
    bmp180RequestTime = millis();
    bmp180State = Bmp180State::REQUESTED_PRESSURE;
  }
}

void BMP180::statePressure() {
  if (millis() - bmp180RequestTime >= 26) {

    // Read the raw pressure value
    // int32_t up = (readWord(wire, BMP180_ADDRESS, BMP180_DATA_REGISTER) << 8) | readByte(wire, BMP180_ADDRESS, BMP180_DATA_REGISTER + 2);

    uint16_t msb = readWord(wire, BMP180_ADDRESS, BMP180_DATA_REGISTER);
    uint8_t xlsb = readByte(wire, BMP180_ADDRESS, BMP180_DATA_REGISTER + 2);

    int32_t up = ((int32_t)msb << 8) | xlsb;
    up >>= (8 - oversampling);  // replace OVERSAMPLING_SETTING with your oversampling setting

    // Calculate the intermediate value b6
    int32_t b6 = b5 - 4000;

    // Calculate intermediate values x1, x2, x3
    int32_t x1 = ((int32_t)b2 * ((b6 * b6) >> 12)) >> 11;
    int32_t x2 = ((int32_t)ac2 * b6) >> 11;
    int32_t x3 = x1 + x2;

    // Calculate the intermediate value b3
    int32_t b3 = ((((int32_t)ac1 * 4 + x3) << oversampling) + 2) / 4;

    // Calculate new intermediate values x1, x2, x3
    x1 = ((int32_t)ac3 * b6) >> 13;
    x2 = ((int32_t)b1 * ((b6 * b6) >> 12)) >> 16;
    x3 = ((x1 + x2) + 2) >> 2;
    uint32_t b4 = ((uint32_t)ac4 * (uint32_t)(x3 + 32768)) >> 15;
    uint32_t b7 = ((uint32_t)up - b3) * (uint32_t)(50000UL >> oversampling);

    // Calculate the pressure value
    if (b7 < 0x80000000) {
        pressure = (b7 * 2) / b4;
    } else {
        pressure = (b7 / b4) * 2;
    }

    // Apply final corrections to the pressure value
    x1 = (pressure >> 8) * (pressure >> 8);
    x1 = (x1 * 3038) >> 16;
    x2 = (-7357 * pressure) >> 16;
    pressure += (x1 + x2 + 3791) >> 4;

    // Calculate altitude
    altitude = 44330.0 * (1.0 - pow(pressure / 101900.0, 0.1903)); // Altitude in m

    bmp180State = Bmp180State::IDLE;
    bmp180RequestTime = millis();
    newData = true;
  }
}

void BMP180::fetchData () {
    switch (this->getState()) {
      case BMP180::Bmp180State::IDLE:
        this->stateIdle();
        break;
      case BMP180::Bmp180State::REQUESTED_TEMP:
        this->stateTemperature();
        break;
      case BMP180::Bmp180State::REQUESTED_PRESSURE:
        this->statePressure();
        break;
  }
}

void BMP180::forceFetchData() {
  newData = false;
  bmp180State = Bmp180State::IDLE;
  
  while (!newData) {
    switch (bmp180State) {
      case Bmp180State::IDLE:
        this->stateIdle();
        break;

      case Bmp180State::REQUESTED_TEMP:
        if (millis() - bmp180RequestTime >= 5) { 
          this->stateTemperature();
        }
        break;

      // Requested Pressure
      case Bmp180State::REQUESTED_PRESSURE:
        if (millis() - bmp180RequestTime >= 26) {
          this->statePressure();
        }
        break;
    }
  }
}