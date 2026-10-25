#include <cassert>

#include "../arduino_smart_glasses.ino/touch_gestures.h"

int main() {
  const TouchGesture none = TouchGesture::none;

  TouchGestureDetector single;
  assert(single.update(false, 0) == none);
  assert(single.update(true, 0) == none);
  assert(single.update(true, 30) == none);
  assert(single.update(false, 120) == none);
  assert(single.update(false, 150) == none);
  assert(single.update(false, 500) == TouchGesture::single);
  assert(single.update(false, 501) == none);

  TouchGestureDetector twice;
  assert(twice.update(true, 0) == none);
  assert(twice.update(true, 30) == none);
  assert(twice.update(false, 100) == none);
  assert(twice.update(false, 130) == none);
  assert(twice.update(true, 200) == none);
  assert(twice.update(true, 230) == none);
  assert(twice.update(false, 280) == none);
  assert(twice.update(false, 310) == TouchGesture::double_tap);
  assert(twice.update(false, 700) == none);

  TouchGestureDetector held;
  assert(held.update(true, 0) == none);
  assert(held.update(true, 30) == none);
  assert(held.update(true, 830) == TouchGesture::long_press);
  assert(held.update(true, 900) == none);
  assert(held.update(false, 950) == none);
  assert(held.update(false, 980) == none);

  TouchGestureDetector bounce;
  assert(bounce.update(true, 0) == none);
  assert(bounce.update(false, 10) == none);
  assert(bounce.update(false, 500) == none);
}
