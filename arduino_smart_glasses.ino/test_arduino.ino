#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// The glasses talk to the laptop over the USB cable at 115200 baud:
//   out  SNAPSHOT, CAM_ON, CAM_OFF   one touch event per line
//   in   PROFILE:... or STATUS:...   what to show for the person in frame
//        CLEAR                       nobody in frame: blank the screen
// The screen only ever shows the person in front of you; hub state
// (recording, saving) is printed in the laptop terminal instead.

// ESP32-S3 touch input and 128x64 I2C OLED (SSD1306, address 0x3C).
const int TOUCH_PIN = 5;
const unsigned long DEBOUNCE_MS = 30;  // a loose jumper chatters faster than this
// The touch module is in toggle mode (its A pad is bridged): every touch
// flips its output and it stays there. So a touch is a flip, not a press,
// and there is no hold. One touch identifies; two within this window start
// or stop recording.
const unsigned long DOUBLE_TOUCH_MS = 600;
const int SDA_PIN = 6;
const int SCL_PIN = 7;
const int SCREEN_WIDTH = 128;
const int SCREEN_HEIGHT = 64;
const int MAX_LINES = 32;
const int CHARS_PER_LINE = 21;
const unsigned long SCROLL_INTERVAL_MS = 160;  // one pixel at a time
const unsigned long TOP_PAUSE_MS = 2500;
const unsigned long BOTTOM_PAUSE_MS = 5000;
const unsigned long OLED_KEEPALIVE_MS = 1000;

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);
bool displayReady = false;
String lines[MAX_LINES];
int lineCount = 0;
int scrollY = 0;
unsigned long lastScrollMs = 0;
unsigned long pauseUntilMs = 0;
bool hasProfile = false;
bool systemActive = false;
unsigned long lastKeepaliveMs = 0;
bool rawTouch = false;
bool stableTouch = false;
bool lastStableTouch = false;
unsigned long rawChangedMs = 0;
bool touchPending = false;
unsigned long firstTouchMs = 0;

void addWrapped(const String& label, const String& value) {
  String remaining = label + value;
  while (remaining.length() && lineCount < MAX_LINES) {
    if (remaining.length() <= CHARS_PER_LINE) {
      lines[lineCount++] = remaining;
      break;
    }
    int split = remaining.lastIndexOf(' ', CHARS_PER_LINE);
    if (split <= 0) split = CHARS_PER_LINE;
    lines[lineCount++] = remaining.substring(0, split);
    remaining = remaining.substring(split);
    remaining.trim();
  }
}

void drawPage() {
  if (!displayReady) return;
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  for (int i = 0; i < lineCount; ++i) {
    int y = i * 9 - scrollY;
    if (y > -8 && y < SCREEN_HEIGHT) {
      display.setCursor(0, y);
      display.print(lines[i]);
    }
  }
  display.display();
}

void showStatus(const String& message) {
  hasProfile = false;
  lineCount = 0;
  scrollY = 0;
  addWrapped("", message);
  drawPage();
}

void showProfile(const String& payload) {
  int separators[3];
  int from = 0;
  for (int i = 0; i < 3; ++i) {
    separators[i] = payload.indexOf('\t', from);
    if (separators[i] < 0) return;  // incomplete serial message
    from = separators[i] + 1;
  }
  lineCount = 0;
  scrollY = 0;
  addWrapped("Name: ", payload.substring(0, separators[0]));
  addWrapped("Met: ", payload.substring(separators[0] + 1, separators[1]));
  addWrapped("Where: ", payload.substring(separators[1] + 1, separators[2]));
  addWrapped("Fact: ", payload.substring(separators[2] + 1));
  hasProfile = true;
  pauseUntilMs = millis() + TOP_PAUSE_MS;
  lastScrollMs = millis();
  drawPage();
}

// A brief dip on a loose VCC jumper resets the SSD1306 to its power-on
// state: display off, charge pump off, image mirrored. It still answers on
// I2C, so nothing notices. Re-sending the settings that differ from that
// state brings it back; on a healthy screen they change nothing, so there is
// no flicker.
void keepDisplayAlive() {
  Wire.beginTransmission(0x3C);
  bool answers = Wire.endTransmission() == 0;
  if (!answers) return;  // unplugged right now; try again next time
  if (!displayReady) displayReady = display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  if (!displayReady) return;
  const uint8_t wake[] = {
      SSD1306_CHARGEPUMP, 0x14,      // power the panel from the internal pump
      SSD1306_MEMORYMODE, 0x00,      // horizontal addressing, as display() expects
      SSD1306_SEGREMAP | 0x1,        // not mirrored
      SSD1306_COMSCANDEC,            // not upside down
      SSD1306_DISPLAYON,
  };
  for (uint8_t command : wake) display.ssd1306_command(command);
  drawPage();
}

void handleMessage(String message) {
  message.trim();
  if (message.startsWith("PROFILE:")) {
    showProfile(message.substring(8));
  } else if (message.startsWith("STATUS:")) {
    showStatus(message.substring(7));
  } else if (message == "CLEAR") {
    showStatus("");
  }
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(50);
  // A loose or missing touch wire reads as "not touched" instead of floating
  // into phantom taps.
  pinMode(TOUCH_PIN, INPUT_PULLDOWN);
  // Whatever state the module woke up in is not a touch.
  rawTouch = stableTouch = lastStableTouch = digitalRead(TOUCH_PIN) == HIGH;
  Wire.begin(SDA_PIN, SCL_PIN);
  // Keep going without the OLED so touch still works, and say why the
  // screen is blank. display.begin() succeeds even with nothing on the bus,
  // so ask the bus first.
  Wire.beginTransmission(0x3C);
  bool oledAnswers = Wire.endTransmission() == 0;
  Serial.println(oledAnswers ? "OLED_FOUND 0x3C" : "OLED_NOT_FOUND 0x3C");
  displayReady = oledAnswers && display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  if (oledAnswers && !displayReady) Serial.println("OLED_INIT_FAILED");

  showStatus("");  // blank until someone is in frame
}

void loop() {
  unsigned long now = millis();
  bool raw = digitalRead(TOUCH_PIN) == HIGH;
  if (raw != rawTouch) {
    rawTouch = raw;
    rawChangedMs = now;
  } else if (now - rawChangedMs >= DEBOUNCE_MS) {
    stableTouch = raw;
  }
  if (stableTouch != lastStableTouch) {
    lastStableTouch = stableTouch;
    if (touchPending && now - firstTouchMs <= DOUBLE_TOUCH_MS) {
      touchPending = false;
      systemActive = !systemActive;
      Serial.println(systemActive ? "CAM_ON" : "CAM_OFF");
    } else {
      touchPending = true;
      firstTouchMs = now;
    }
  }
  if (touchPending && now - firstTouchMs > DOUBLE_TOUCH_MS) {
    touchPending = false;
    Serial.println("SNAPSHOT");
  }

  if (Serial.available()) {
    handleMessage(Serial.readStringUntil('\n'));
  }

  if (now - lastKeepaliveMs >= OLED_KEEPALIVE_MS) {
    lastKeepaliveMs = now;
    keepDisplayAlive();
  }

  if (hasProfile && now >= pauseUntilMs && now - lastScrollMs >= SCROLL_INTERVAL_MS) {
    lastScrollMs = now;
    int bottom = max(0, lineCount * 9 - SCREEN_HEIGHT);
    if (scrollY < bottom) {
      ++scrollY;
      drawPage();
    } else {
      scrollY = 0;
      pauseUntilMs = now + BOTTOM_PAUSE_MS;
      drawPage();
    }
  }
}
