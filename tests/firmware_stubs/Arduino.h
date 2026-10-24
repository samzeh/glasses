#pragma once

#include <stdint.h>

#include <string>

#define HIGH 1
#define INPUT 0

class String {
 public:
  explicit String(const char* value = "") : value_(value) {}
  void trim() {}
  size_t length() const { return value_.length(); }

 private:
  std::string value_;
};

struct FakeSerial {
  std::string output;
  unsigned long baud = 0;
  void begin(unsigned long speed) { baud = speed; }
  void print(const char* value) { output += value; }
  void print(unsigned long value) { output += std::to_string(value); }
  void println(const char* value) { output += std::string(value) + "\n"; }
  int available() { return 0; }
  String readStringUntil(char) { return String(); }
};

extern FakeSerial Serial;
void pinMode(int, int);
int digitalRead(int);
unsigned long millis();
unsigned long micros();
void delay(unsigned long);
void yield();
