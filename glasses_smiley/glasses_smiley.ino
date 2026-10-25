// Smart glasses display test - ESP32-S3 + SSD1306 I2C OLED at 0x3C
// Draws a large smiley face, keeps the touch -> serial behaviour from
// test_arduino.ino. No external library: talks to the OLED over Wire using
// page addressing (works on SSD1306 and SH1106 panels).

#include <Wire.h>

// Pins (same as test_arduino.ino)
const int TOUCH_PIN = 4;
const int SDA_PIN   = 6;
const int SCL_PIN   = 7;

const uint8_t OLED_W = 128;
const uint8_t OLED_H = 64;          // set to 32 for a 0.91" 128x32 panel
uint8_t oledAddr = 0x3C;

uint8_t fb[OLED_W * OLED_H / 8];    // 1 bit per pixel, 8 rows per page

// ---------- low level OLED ----------
void oledCmd(uint8_t c) {
  Wire.beginTransmission(oledAddr);
  Wire.write(0x00);
  Wire.write(c);
  Wire.endTransmission();
}

void oledCmd2(uint8_t c, uint8_t v) {
  Wire.beginTransmission(oledAddr);
  Wire.write(0x00);
  Wire.write(c);
  Wire.write(v);
  Wire.endTransmission();
}

void oledInit() {
  delay(100);
  oledCmd(0xAE);                      // display off
  oledCmd2(0xD5, 0x80);               // clock divide
  oledCmd2(0xA8, OLED_H - 1);         // multiplex ratio
  oledCmd2(0xD3, 0x00);               // display offset
  oledCmd(0x40);                      // start line 0
  oledCmd2(0x8D, 0x14);               // charge pump on
  oledCmd2(0x20, 0x02);               // page addressing mode
  oledCmd(0xA1);                      // segment remap
  oledCmd(0xC8);                      // COM scan direction reversed
  oledCmd2(0xDA, OLED_H == 64 ? 0x12 : 0x02);   // COM pin config
  oledCmd2(0x81, 0xFF);               // contrast - maximum
  oledCmd2(0xD9, 0xF1);               // pre-charge - longest, brightest
  oledCmd2(0xDB, 0x00);               // Vcomh deselect 0.65*Vcc - brightest
  oledCmd(0xA4);                      // display follows RAM
  oledCmd(0xA6);                      // non-inverted
  oledCmd(0xAF);                      // display on
}

void oledFlush() {
  for (uint8_t page = 0; page < OLED_H / 8; page++) {
    oledCmd(0xB0 | page);
    oledCmd(0x00);                    // column low nibble
    oledCmd(0x10);                    // column high nibble
    for (uint8_t x = 0; x < OLED_W; x += 16) {
      Wire.beginTransmission(oledAddr);
      Wire.write(0x40);               // data stream
      for (uint8_t i = 0; i < 16; i++) Wire.write(fb[page * OLED_W + x + i]);
      Wire.endTransmission();
    }
  }
}

void fbClear() { memset(fb, 0, sizeof(fb)); }

void setPixel(int x, int y) {
  if (x < 0 || y < 0 || x >= OLED_W || y >= OLED_H) return;
  fb[(y / 8) * OLED_W + x] |= (1 << (y % 8));
}

// ---------- the smiley ----------
void drawSmiley() {
  fbClear();

  const float cx = OLED_W / 2.0f - 0.5f;          // 63.5
  const float cy = OLED_H / 2.0f - 0.5f;          // 31.5
  const float faceR = (OLED_H / 2.0f) - 2.0f;     // 30
  const float eyeR  = faceR * 0.135f;
  const float eyeDX = faceR * 0.37f;
  const float eyeDY = faceR * 0.33f;
  const float mouthR = faceR * 0.60f;
  const float mouthCY = cy - faceR * 0.13f;

  for (int y = 0; y < OLED_H; y++) {
    for (int x = 0; x < OLED_W; x++) {
      float dx = x - cx, dy = y - cy;
      float d = sqrtf(dx * dx + dy * dy);

      // face outline
      if (fabsf(d - faceR) < 1.6f) { setPixel(x, y); continue; }

      // eyes (filled)
      float le = sqrtf((x - (cx - eyeDX)) * (x - (cx - eyeDX)) +
                       (y - (cy - eyeDY)) * (y - (cy - eyeDY)));
      float re = sqrtf((x - (cx + eyeDX)) * (x - (cx + eyeDX)) +
                       (y - (cy - eyeDY)) * (y - (cy - eyeDY)));
      if (le <= eyeR || re <= eyeR) { setPixel(x, y); continue; }

      // smile: lower arc only
      float md = sqrtf(dx * dx + (y - mouthCY) * (y - mouthCY));
      if (fabsf(md - mouthR) < 1.8f && y > mouthCY + mouthR * 0.42f) setPixel(x, y);
    }
  }
  oledFlush();
}

// 5x7 font, just the characters the status overlays need
const char *GLYPH_CHARS = "ABCDEFGHIKLMNOPRSTUV_ ";
const uint8_t GLYPHS[][5] = {
  {0x7E,0x11,0x11,0x11,0x7E}, {0x7F,0x49,0x49,0x49,0x36}, {0x3E,0x41,0x41,0x41,0x22},
  {0x7F,0x41,0x41,0x22,0x1C}, {0x7F,0x49,0x49,0x49,0x41}, {0x7F,0x09,0x09,0x09,0x01},
  {0x3E,0x41,0x49,0x49,0x7A}, {0x7F,0x08,0x08,0x08,0x7F}, {0x00,0x41,0x7F,0x41,0x00},
  {0x7F,0x08,0x14,0x22,0x41}, {0x7F,0x40,0x40,0x40,0x40}, {0x7F,0x02,0x0C,0x02,0x7F},
  {0x7F,0x04,0x08,0x10,0x7F}, {0x3E,0x41,0x41,0x41,0x3E}, {0x7F,0x09,0x09,0x09,0x06},
  {0x7F,0x09,0x19,0x29,0x46}, {0x46,0x49,0x49,0x49,0x31}, {0x01,0x01,0x7F,0x01,0x01},
  {0x3F,0x40,0x40,0x40,0x3F}, {0x1F,0x20,0x40,0x20,0x1F}, {0x40,0x40,0x40,0x40,0x40},
  {0x00,0x00,0x00,0x00,0x00},
};

void drawChar(int x, int y, char c) {
  const char *p = strchr(GLYPH_CHARS, c);
  if (!p) p = strchr(GLYPH_CHARS, ' ');
  const uint8_t *g = GLYPHS[p - GLYPH_CHARS];
  for (int col = 0; col < 5; col++)
    for (int row = 0; row < 7; row++)
      if (g[col] & (1 << row)) setPixel(x + col, y + row);
}

void drawText(int x, int y, const char *s) {
  while (*s) { drawChar(x, y, *s++); x += 6; }
}

void showMessage(const char *text) {
  drawSmiley();                        // keep the face, label it underneath
  int w = strlen(text) * 6;
  int x = (OLED_W - w) / 2;
  for (int yy = OLED_H - 10; yy < OLED_H; yy++)
    for (int xx = x - 3; xx < x + w + 2; xx++)
      if (xx >= 0 && xx < OLED_W) fb[(yy / 8) * OLED_W + xx] &= ~(1 << (yy % 8));
  drawText(x, OLED_H - 9, text);
  oledFlush();
}

// ---------- serial preview so the panel contents are verifiable ----------
void dumpFrame() {
  Serial.println("--- framebuffer (2x downsampled) ---");
  for (int y = 0; y < OLED_H; y += 2) {
    char line[OLED_W / 2 + 1];
    int n = 0;
    for (int x = 0; x < OLED_W; x += 2) {
      bool on = (fb[(y / 8) * OLED_W + x] & (1 << (y % 8))) ||
                (fb[((y + 1) / 8) * OLED_W + x] & (1 << ((y + 1) % 8)));
      line[n++] = on ? '#' : '.';
    }
    line[n] = 0;
    Serial.println(line);
  }
  Serial.println("--- end framebuffer ---");
}

// ---------- touch state ----------
bool systemActive = false;
unsigned long touchStartTime = 0;
bool isTouching = false;
bool holdTriggered = false;
unsigned long messageUntil = 0;

int sdaUsed = SDA_PIN;
int sclUsed = SCL_PIN;

// Brings up the bus and finds the display, tolerating SDA/SCL being wired
// either way round - a swapped pair of jumpers is otherwise a silent failure.
uint8_t bringUpBus() {
  const int orders[2][2] = {{SDA_PIN, SCL_PIN}, {SCL_PIN, SDA_PIN}};
  for (int o = 0; o < 2; o++) {
    Wire.end();
    Wire.begin(orders[o][0], orders[o][1], 400000);
    Serial.printf("I2C scan on SDA=%d SCL=%d:\n", orders[o][0], orders[o][1]);
    for (uint8_t a = 1; a < 127; a++) {
      Wire.beginTransmission(a);
      if (Wire.endTransmission() == 0) {
        sdaUsed = orders[o][0];
        sclUsed = orders[o][1];
        Serial.printf("  device at 0x%02X%s\n", a,
                      o ? "  (SDA/SCL wired opposite to the sketch default)" : "");
        return a;
      }
    }
    Serial.println("  nothing found");
  }
  Serial.println("no device on either pin order - check OLED power and wiring");
  Wire.end();
  Wire.begin(SDA_PIN, SCL_PIN, 400000);
  return 0x3C;
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println();
  Serial.printf("=== glasses smiley test (ESP32-S3, %dx%d OLED) ===\n", OLED_W, OLED_H);

  pinMode(TOUCH_PIN, INPUT);

  oledAddr = bringUpBus();
  Serial.printf("driving OLED at 0x%02X on SDA=%d SCL=%d\n", oledAddr, sdaUsed, sclUsed);

  oledInit();
  Serial.println("brightness: contrast=0xFF (max), precharge=0xF1, Vcomh=0x00");
  drawSmiley();
  dumpFrame();
  Serial.println("smiley on screen - tap for SNAPSHOT, hold 800ms to toggle CAM");
}

void loop() {
  int currentTouchState = digitalRead(TOUCH_PIN);
  unsigned long currentMillis = millis();

  if (currentTouchState == HIGH) {
    if (!isTouching) {
      isTouching = true;
      touchStartTime = currentMillis;
      holdTriggered = false;
    } else if (!holdTriggered && (currentMillis - touchStartTime >= 800)) {
      holdTriggered = true;
      systemActive = !systemActive;
      Serial.println(systemActive ? "CAM_ON" : "CAM_OFF");
      showMessage(systemActive ? "CAM_ON" : "CAM_OFF");
      messageUntil = currentMillis + 1500;
    }
  } else if (isTouching) {
    unsigned long touchDuration = currentMillis - touchStartTime;
    if (!holdTriggered && touchDuration >= 50 && touchDuration < 800) {
      Serial.println("SNAPSHOT");
      showMessage("SNAPSHOT");
      messageUntil = currentMillis + 1500;
    }
    isTouching = false;
  }

  // status updates coming back from Python over serial
  if (Serial.available() > 0) {
    String pythonMsg = Serial.readStringUntil('\n');
    pythonMsg.trim();
    pythonMsg.toUpperCase();
    if (pythonMsg.length() > 0) {
      char buf[22];
      pythonMsg.toCharArray(buf, sizeof(buf));
      showMessage(buf);
      messageUntil = currentMillis + 2000;
    }
  }

  if (messageUntil && currentMillis > messageUntil) {
    messageUntil = 0;
    drawSmiley();
  }

  delay(10);
}
