#ifndef I2CDEV_H
#define I2CDEV_H

#include <Arduino.h>
#include <Wire.h>

namespace I2Cdev {
  byte readByte(TwoWire& wire, byte addr, byte reg);
  uint16_t readWord(TwoWire& wire, byte addr, byte reg);
  byte writeData(TwoWire& wire, byte addr, byte reg, byte data);

  // quantity is number of words to read
  void readWords(TwoWire& wire, byte addr, byte reg, uint16_t* data, int quantity);

}

#endif
