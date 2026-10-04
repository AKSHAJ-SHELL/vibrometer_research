// Arduino Uno power meter for the ESP32-S3 (see firmware/README.md for wiring).
//
// A0 = voltage across the shunt in the ESP32's ground return (current = V / R_shunt).
// A1 = ESP32 marker pin (3.3 V high). Read as analog with a 512 threshold, so the 3.3 V
//      signal is unambiguous on a 5 V board and the ground offset across the shunt is ignored.
// Reference: internal 1.1 V (about 1.07 mV per step). Its exact value varies ±10% between
// boards — measure it on the AREF pin, or calibrate with a known resistor (README).
//
// Output, 500000 baud, one line per 1 ms block:  t_us,adc_x100,marker,n
//   t_us      micros() at block start (wraps every ~71 min; the capture script unwraps it)
//   adc_x100  mean A0 reading × 100 over the block
//   marker    1 if the marker was high for most of the block
//   n         readings in the block

const uint8_t SHUNT_PIN = A0;
const uint8_t MARKER_PIN = A1;

void setup() {
  Serial.begin(500000);
  analogReference(INTERNAL);
  for (int i = 0; i < 200; ++i) analogRead(SHUNT_PIN);   // let the reference settle
  Serial.println("t_us,adc_x100,marker,n");
}

void loop() {
  const uint32_t start = micros();
  uint32_t sum = 0;
  uint16_t n = 0, high = 0;
  while (micros() - start < 1000UL) {
    sum += analogRead(SHUNT_PIN);
    high += analogRead(MARKER_PIN) > 512;
    ++n;
  }
  Serial.print(start);
  Serial.print(',');
  Serial.print(sum * 100UL / n);
  Serial.print(',');
  Serial.print(high * 2 > n ? 1 : 0);
  Serial.print(',');
  Serial.println(n);
}
