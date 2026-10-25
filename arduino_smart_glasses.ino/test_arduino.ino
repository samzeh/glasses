#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ESP32 touch input and 128x64 I2C OLED (SSD1306, address 0x3C).
const int TOUCH_PIN = 4;
const int SDA_PIN = 6;
const int SCL_PIN = 7;
const int SCREEN_WIDTH = 128;
const int SCREEN_HEIGHT = 64;
const int MAX_LINES = 32;
const int CHARS_PER_LINE = 21;
const unsigned long SCROLL_INTERVAL_MS = 160;  // one pixel at a time
const unsigned long TOP_PAUSE_MS = 2500;
const unsigned long BOTTOM_PAUSE_MS = 5000;

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);
String lines[MAX_LINES];
int lineCount = 0;
int scrollY = 0;
unsigned long lastScrollMs = 0;
unsigned long pauseUntilMs = 0;
bool hasProfile = false;
bool systemActive = false;
unsigned long touchStartTime = 0;
bool isTouching = false;
bool holdTriggered = false;

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

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(50);
  pinMode(TOUCH_PIN, INPUT);
  Wire.begin(SDA_PIN, SCL_PIN);
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("OLED_INIT_FAILED");
    while (true) delay(1000);
  }
  showStatus("Glasses ready");
}

void loop() {
  unsigned long now = millis();
  bool touching = digitalRead(TOUCH_PIN) == HIGH;
  if (touching && !isTouching) {
    isTouching = true;
    touchStartTime = now;
    holdTriggered = false;
  } else if (touching && !holdTriggered && now - touchStartTime >= 800) {
    holdTriggered = true;
    systemActive = !systemActive;
    showStatus(systemActive ? "Recording..." : "Saving meeting...");
    Serial.println(systemActive ? "CAM_ON" : "CAM_OFF");
  } else if (!touching && isTouching) {
    if (!holdTriggered && now - touchStartTime >= 50) {
      Serial.println("SNAPSHOT");
      showStatus("Looking...");
    }
    isTouching = false;
  }

  if (Serial.available()) {
    String message = Serial.readStringUntil('\n');
    message.trim();
    if (message.startsWith("PROFILE:")) {
      showProfile(message.substring(8));
    } else if (message.startsWith("STATUS:")) {
      showStatus(message.substring(7));
    }
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
