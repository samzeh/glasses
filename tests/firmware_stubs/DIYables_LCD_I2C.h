#pragma once

class DIYables_LCD_I2C {
 public:
  DIYables_LCD_I2C(int, int, int) {}
  void init() {}
  void backlight() {}
  void clear() {}
  void setCursor(int, int) {}
  template <typename T>
  void print(const T&) {}
};
