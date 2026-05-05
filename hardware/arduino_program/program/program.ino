/*
  ESP32 + L298N Bluetooth drive
  SIMILAR APPROACH TO lectures_examples_prof/bluetooth_robot_1.1_vision

  Wiring :
    GPIO23 -> IN1   Motor A (LEFT)
    GPIO22 -> IN2
    GPIO33 -> IN3   Motor B (RIGHT)
    GPIO25 -> IN4
    GND    -> L298N GND
    ENA / ENB jumpered ON (always full speeddd)

*/

#include "BluetoothSerial.h"

#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled. In Arduino IDE: Tools -> select an ESP32 board, not ESP32-S3/C3.
#endif

#define IN1 23
#define IN2 22
#define IN3 33
#define IN4 25

BluetoothSerial SerialBT;

// servo-style command range
// 90 = stop, >90 = forward, <90 = backward, 
static const int TH_STOP     = 90;
static const int TH_DEADBAND = 5;

void stop_motors() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}

// drive one side of an L298N from a 0..180 servo-style command
void drive_side(int th, int in_a, int in_b) {
  if (th > TH_STOP + TH_DEADBAND) {
    digitalWrite(in_a, HIGH);
    digitalWrite(in_b, LOW);
  } else if (th < TH_STOP - TH_DEADBAND) {
    digitalWrite(in_a, LOW);
    digitalWrite(in_b, HIGH);
  } else {
    digitalWrite(in_a, LOW);
    digitalWrite(in_b, LOW);
  }
}

void write_actuators(int th_left, int th_right, int /*th_aux*/) {
  if (th_left  < 0)   th_left  = 0;
  if (th_left  > 180) th_left  = 180;
  if (th_right < 0)   th_right = 0;
  if (th_right > 180) th_right = 180;

  drive_side(th_left,  IN1, IN2);
  drive_side(th_right, IN3, IN4);
}

void setup() {
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  stop_motors();

  Serial.begin(115200);   // USB serial, debug only
  SerialBT.begin("ESP32_Robot2");  // pair name on Windows

  Serial.println("ESP32 BT robot ready. Pair with 'ESP32_Robot'.");

  while (SerialBT.available() == 0) {
    delay(1);
  }
  SerialBT.read();  // consume the start byte
  Serial.println("PC handshake received, entering loop.");
}

void loop() {
  static char buffer_in[4];
  const unsigned char start_char = 255;
  const unsigned long TIMEOUT_MS = 1000;  // matches prof's setTimeout(1000)

  // wait for start byte
  unsigned long t0 = millis();
  while (true) {
    if (SerialBT.available() > 0) {
      unsigned char b = (unsigned char)SerialBT.read();
      if (b == start_char) break;
      // any non-start byte: keep scanning (resync, same idea as prof's)
    }
    if (millis() - t0 > TIMEOUT_MS) {
      stop_motors();   // comms-fault safety
      t0 = millis();
    }
  }

  // read the 3 data bytes
  int got = 0;
  t0 = millis();
  while (got < 3) {
    if (SerialBT.available() > 0) {
      buffer_in[got++] = SerialBT.read();
    }
    if (millis() - t0 > TIMEOUT_MS) {
      stop_motors();
      return;  // bail and resync at the top of loop()
    }
  }

  int th_left  = (unsigned char)buffer_in[0];
  int th_right = (unsigned char)buffer_in[1];
  int th_aux   = (unsigned char)buffer_in[2];

  write_actuators(th_left, th_right, th_aux);
}
