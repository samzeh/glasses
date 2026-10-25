// I2C diagnostic for the smart glasses OLED - ESP32-S3
// 1. reports idle levels on the expected SDA/SCL pins
// 2. full address scan on SDA=6 / SCL=7
// 3. sweeps other GPIO pairs looking for a display at 0x3C/0x3D/0x27/0x3F

#include <Wire.h>

const int SDA_PIN = 6;
const int SCL_PIN = 7;

// Safe to probe on this board. Deliberately excluded:
// 19/20 = USB D-/D+, 26-32 = SPI flash, 33-37 = octal PSRAM, 43/44 = UART0.
const int CANDIDATES[] = {1,2,3,4,5,8,9,10,11,12,13,14,15,16,17,18,21,38,39,40,41,42,45,46,47,48,6,7};
const int NCAND = sizeof(CANDIDATES) / sizeof(CANDIDATES[0]);
const uint8_t DISPLAY_ADDRS[] = {0x3C, 0x3D, 0x27, 0x3F};

void reportLevels() {
  Serial.println("1) idle line levels on the expected pins");

  pinMode(SDA_PIN, INPUT);
  pinMode(SCL_PIN, INPUT);
  delay(5);
  int sdaFloat = digitalRead(SDA_PIN);
  int sclFloat = digitalRead(SCL_PIN);

  pinMode(SDA_PIN, INPUT_PULLUP);
  pinMode(SCL_PIN, INPUT_PULLUP);
  delay(5);
  int sdaPull = digitalRead(SDA_PIN);
  int sclPull = digitalRead(SCL_PIN);

  Serial.printf("   GPIO%d (SDA): no-pull=%s  internal-pullup=%s\n",
                SDA_PIN, sdaFloat ? "HIGH" : "LOW", sdaPull ? "HIGH" : "LOW");
  Serial.printf("   GPIO%d (SCL): no-pull=%s  internal-pullup=%s\n",
                SCL_PIN, sclFloat ? "HIGH" : "LOW", sclPull ? "HIGH" : "LOW");

  if (!sdaPull || !sclPull) {
    Serial.println("   >> a line is stuck LOW even with the pull-up on:");
    Serial.println("      shorted to GND, or a powered-down device clamping the bus");
  } else if (sdaFloat && sclFloat) {
    Serial.println("   >> both lines sit HIGH unconnected: external pull-ups present,");
    Serial.println("      so the module is powered - but it is not answering");
  } else {
    Serial.println("   >> lines float LOW with no pull-up: no external pull-ups seen,");
    Serial.println("      which usually means the module is unpowered or unplugged");
  }
  pinMode(SDA_PIN, INPUT);
  pinMode(SCL_PIN, INPUT);
}

int scanBus(int sda, int scl, bool verbose) {
  Wire.end();
  if (!Wire.begin(sda, scl, 100000)) return 0;
  int hits = 0;
  if (verbose) {
    for (uint8_t a = 1; a < 127; a++) {
      Wire.beginTransmission(a);
      if (Wire.endTransmission() == 0) {
        Serial.printf("   ACK at 0x%02X\n", a);
        hits++;
      }
    }
  } else {
    for (uint8_t i = 0; i < sizeof(DISPLAY_ADDRS); i++) {
      Wire.beginTransmission(DISPLAY_ADDRS[i]);
      if (Wire.endTransmission() == 0) {
        Serial.printf("   ACK at 0x%02X on SDA=%d SCL=%d  <<< FOUND\n",
                      DISPLAY_ADDRS[i], sda, scl);
        hits++;
      }
    }
  }
  return hits;
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println();
  Serial.println("=== OLED I2C diagnostic (ESP32-S3) ===");

  reportLevels();

  Serial.printf("\n2) full address scan on SDA=%d SCL=%d\n", SDA_PIN, SCL_PIN);
  int found = scanBus(SDA_PIN, SCL_PIN, true);
  if (!found) Serial.println("   no ACK from any of the 126 addresses");

  Serial.println("\n3) sweeping other GPIO pairs for a display address");
  int swept = 0, hits = 0;
  for (int i = 0; i < NCAND; i++) {
    for (int j = 0; j < NCAND; j++) {
      if (i == j) continue;
      if (CANDIDATES[i] == SDA_PIN && CANDIDATES[j] == SCL_PIN) continue;
      hits += scanBus(CANDIDATES[i], CANDIDATES[j], false);
      swept++;
    }
  }
  Serial.printf("   swept %d pin pairs, %d hit(s)\n", swept, hits);
  if (!hits) Serial.println("   >> the display is not responding on ANY pin pair");

  Serial.println("\n=== diagnostic complete ===");
}

void loop() {
  delay(1000);
}
