#include <Wire.h>
#include <DIYables_LCD_I2C.h>

// Pins
const int TOUCH_PIN = 4;
const int SDA_PIN = 6;
const int SCL_PIN = 7;

DIYables_LCD_I2C lcd(0x27, 16, 2);

// State tracking variables
bool systemActive = false;
unsigned long touchStartTime = 0;
bool isTouching = false;
bool holdTriggered = false;

void setup() {
  Serial.begin(115200);
  delay(1500);

  pinMode(TOUCH_PIN, INPUT);

  Wire.begin(SDA_PIN, SCL_PIN);
  lcd.init();
  lcd.backlight();
  
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Glasses Ready   ");
  lcd.setCursor(0, 1);
  lcd.print("Status: STANDBY ");
}

void loop() {
  int currentTouchState = digitalRead(TOUCH_PIN);
  unsigned long currentMillis = millis();

  if (currentTouchState == HIGH) {
    if (!isTouching) {
      isTouching = true;
      touchStartTime = currentMillis;
      holdTriggered = false;
    } else {
      // Check if held for 800ms or more (Toggle CAM_ON / CAM_OFF)
      if (!holdTriggered && (currentMillis - touchStartTime >= 800)) {
        holdTriggered = true;
        systemActive = !systemActive;

        lcd.clear();
        lcd.setCursor(0, 0);
        if (systemActive) {
          lcd.print("Mode: ACTIVE    ");
          lcd.setCursor(0, 1);
          lcd.print("Sending CAM_ON  ");
          Serial.println("CAM_ON");
        } else {
          lcd.print("Mode: STANDBY   ");
          lcd.setCursor(0, 1);
          lcd.print("Sending CAM_OFF ");
          Serial.println("CAM_OFF");
        }
      }
    }
  } else {
    if (isTouching) {
      unsigned long touchDuration = currentMillis - touchStartTime;

      // Quick tap detection (Trigger SNAPSHOT if released under 800ms)
      if (!holdTriggered && touchDuration >= 50 && touchDuration < 800) {
        lcd.setCursor(0, 1);
        lcd.print("Action: SNAPSHOT");
        Serial.println("SNAPSHOT");
        delay(1000);
        
        lcd.setCursor(0, 1);
        lcd.print(systemActive ? "Status: ACTIVE  " : "Status: STANDBY ");
      }

      isTouching = false;
    }
  }

  // Check for status updates coming back from Python over Serial
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