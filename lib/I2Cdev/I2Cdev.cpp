#include <I2Cdev.h>

namespace I2Cdev {
  byte writeData(TwoWire& wire, byte addr, byte reg, byte data) {
    wire.beginTransmission(addr);
    wire.write(reg);
    wire.write(data);
    byte status = wire.endTransmission();
    return status; // 0 if success
  }

  // This method is not used internally, maybe by user...
  byte readByte(TwoWire& wire, byte addr, byte reg) {
    wire.beginTransmission(addr);
    wire.write(reg);
    wire.endTransmission(true);
    wire.requestFrom(addr, 1);
    byte data = wire.read();
    return data;
  }

  void readWords(TwoWire& wire, byte addr, byte reg, uint16_t* data, int quantity) {
    wire.beginTransmission(addr);
    wire.write(reg);
    wire.endTransmission(true);
    wire.requestFrom(addr, quantity * 2);  // Multiply by 2 to account for 2 bytes per value
    for (int i = 0; i < quantity; i++) {
      data[i] = wire.read() << 8 | wire.read();
    }
  }

  uint16_t readWord(TwoWire& wire, byte addr, byte reg) {
    uint8_t msb = readByte(wire, addr, reg);
    uint8_t lsb = readByte(wire, addr, reg + 1);
    return (msb << 8) | lsb;
  }
}