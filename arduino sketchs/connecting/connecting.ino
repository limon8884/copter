
const byte numSignals = 2;
byte signals[numSignals];
const int delay_time = 20; // in millis

void setup() {
    Serial.begin(115200);
    Serial.setTimeout(1);
}

void loop() {
    while (!Serial.available());
    if (Serial.available())
    {    
        Serial.readBytes(signals, numSignals);
        // send signals to motors

        // get data from env 

        int angle = int(signals[0] + 1);
        int velocity = int(signals[1] + 1);
        int acceleration = int(signals[1] + 10);

        String s = String(angle) + "," + String(velocity) + "," + String(acceleration);
        Serial.print(s);
        delay(delay_time);
    }
}