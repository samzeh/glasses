#include <Wire.h>
#include <DIYables_LCD_I2C.h>
#include "touch_gestures.h"

const int TOUCH_PIN = 4;
const int SDA_PIN = 6;
const int SCL_PIN = 7;
const unsigned long SERIAL_BAUD = 921600;

DIYables_LCD_I2C lcd(0x27, 16, 2);
TouchGestureDetector touch;
bool systemActive = false;

void showStatus(const char* line1, const char* line2) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(line1);
  lcd.setCursor(0, 1);
  lcd.print(line2);
}

void sendTouch(const char* gesture) {
  // Same v1 message read by glasses.device.serial.EspSerialStream.
  Serial.print("{\"v\":1,\"type\":\"touch\",\"t_us\":");
  Serial.print(micros());
  Serial.print(",\"gesture\":\"");
  Serial.print(gesture);
  Serial.println("\"}");
}

void handleGesture(TouchGesture gesture) {
  switch (gesture) {
    case TouchGesture::single:
      sendTouch("single");
      // Preserve the older vision_hub.py command until that path is retired.
      Serial.println("SNAPSHOT");
      showStatus("Glasses Ready", "Action: SINGLE");
      break;
    case TouchGesture::double_tap:
      sendTouch("double");
      showStatus("Glasses Ready", "Action: DOUBLE");
      break;
    case TouchGesture::long_press:
      sendTouch("long");
      systemActive = !systemActive;
      Serial.println(systemActive ? "CAM_ON" : "CAM_OFF");
      showStatus(systemActive ? "Mode: ACTIVE" : "Mode: STANDBY",
                 systemActive ? "Sending CAM_ON" : "Sending CAM_OFF");
      break;
    case TouchGesture::none:
      break;
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(1500);
  pinMode(TOUCH_PIN, INPUT);
  Wire.begin(SDA_PIN, SCL_PIN);
  lcd.init();
  lcd.backlight();
  showStatus("Glasses Ready", "Status: STANDBY");
}

void loop() {
  handleGesture(touch.update(digitalRead(TOUCH_PIN) == HIGH, millis()));

  // The older hub can still send text to the LCD over this serial link.
  if (Serial.available() > 0) {
    String pythonMsg = Serial.readStringUntil('\n');
    pythonMsg.trim();
    if (pythonMsg.length() > 0) {
      lcd.setCursor(0, 1);
      lcd.print("                ");
      lcd.setCursor(0, 1);
      lcd.print(pythonMsg);
    }
  }
  yield();
}
