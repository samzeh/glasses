#pragma once

struct FakeWire {
  void begin(int, int) {}
};

extern FakeWire Wire;
