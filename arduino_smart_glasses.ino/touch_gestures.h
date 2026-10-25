#pragma once

#include <stdint.h>

enum class TouchGesture { none, single, double_tap, long_press };

// Kept independent of Arduino APIs so the gesture timing can be tested on a
// laptop. update() should be called on every pass through loop().
class TouchGestureDetector {
 public:
  TouchGesture update(bool touching, uint32_t now_ms) {
    if (touching != raw_) {
      raw_ = touching;
      raw_changed_at_ = now_ms;
    }

    if (raw_ != stable_ && now_ms - raw_changed_at_ >= debounce_ms) {
      stable_ = raw_;
      if (stable_) {
        pressed_at_ = now_ms;
        long_sent_ = false;
        second_press_ = pending_single_ &&
                        now_ms - first_released_at_ <= double_tap_ms;
      } else if (!long_sent_ && now_ms - pressed_at_ >= min_tap_ms) {
        if (second_press_) {
          pending_single_ = false;
          second_press_ = false;
          return TouchGesture::double_tap;
        }
        pending_single_ = true;
        first_released_at_ = now_ms;
      }
    }

    if (stable_ && !long_sent_ && now_ms - pressed_at_ >= long_press_ms) {
      long_sent_ = true;
      pending_single_ = false;
      second_press_ = false;
      return TouchGesture::long_press;
    }

    if (pending_single_ && !second_press_ &&
        now_ms - first_released_at_ >= double_tap_ms) {
      pending_single_ = false;
      return TouchGesture::single;
    }
    return TouchGesture::none;
  }

 private:
  static constexpr uint32_t debounce_ms = 25;
  static constexpr uint32_t min_tap_ms = 50;
  static constexpr uint32_t double_tap_ms = 350;
  static constexpr uint32_t long_press_ms = 800;

  bool raw_ = false;
  bool stable_ = false;
  bool pending_single_ = false;
  bool second_press_ = false;
  bool long_sent_ = false;
  uint32_t raw_changed_at_ = 0;
  uint32_t pressed_at_ = 0;
  uint32_t first_released_at_ = 0;
};
