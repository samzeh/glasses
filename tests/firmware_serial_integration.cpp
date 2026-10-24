#include <cassert>
#include <iostream>

#include "firmware_stubs/Arduino.h"
#include "firmware_stubs/Wire.h"

FakeSerial Serial;
FakeWire Wire;
static unsigned long now_ms = 0;
static int touch_level = 0;

void pinMode(int, int) {}
int digitalRead(int) { return touch_level; }
unsigned long millis() { return now_ms; }
unsigned long micros() { return now_ms * 1000; }
void delay(unsigned long amount) { now_ms += amount; }
void yield() {}

#include "../arduino_smart_glasses.ino/test_arduino.ino"

void tick(bool touching, unsigned long at_ms) {
  touch_level = touching ? HIGH : 0;
  now_ms = at_ms;
  loop();
}

int main() {
  setup();
  assert(Serial.baud == 921600);

  tick(true, 0);
  tick(true, 30);
  tick(false, 120);
  tick(false, 150);
  tick(false, 500);  // single tap, after the double-tap window

  tick(true, 1000);
  tick(true, 1030);
  tick(false, 1100);
  tick(false, 1130);
  tick(true, 1200);
  tick(true, 1230);
  tick(false, 1280);
  tick(false, 1310);  // double tap

  tick(true, 2000);
  tick(true, 2030);
  tick(true, 2830);  // long press fires while still touching
  tick(false, 2900);
  tick(false, 2930);

  std::cout << Serial.output;
}
