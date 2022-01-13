#define time_control millis

const unsigned int ARRAY_LENGTH = 10;
//const float delay_millis = 500;

const unsigned int pin = 3;
volatile unsigned int count = 0;
volatile unsigned int timer = 0;

volatile boolean process_flag = true;
volatile boolean start_log_flag = false;
volatile boolean stop_log_flag = false;

volatile unsigned int arr[ARRAY_LENGTH] = {0}; 

volatile unsigned int array_position = 0;

void setup()
{
  Serial.begin(9600);    //открываем порт
  pinMode(pin, INPUT);
  attachInterrupt(digitalPinToInterrupt(pin), catch_signal,  FALLING); 
  timer = time_control();
}

void catch_signal()
{
    unsigned int curr_timer = time_control();
    unsigned int delta = curr_timer - timer;
    if (start_log_flag && !stop_log_flag && delta > 1)
    {    
        count++;    
        timer = curr_timer;       
        arr[array_position] += delta;
        if (!process_flag)
            array_position++;
        process_flag = !process_flag;
        if (array_position >= ARRAY_LENGTH) 
        {
           array_position = 0;
           stop_log_flag = true; 
        }
    }
}

void loop() 
{
    if (Serial.available())
    {
        char w = Serial.read();
        if (w == 'T')
        timer = time_control();
        start_log_flag = true;
    }
    if (start_log_flag && stop_log_flag)
    {
       start_log_flag = false;
       stop_log_flag = false;
       process_flag = true;
       Serial.println("######################");
       for (int i = 0; i < ARRAY_LENGTH; i++)
       {
          Serial.println(arr[i]);
          arr[i] = 0;
       }
       Serial.println();
       delay(100);
    }
    Serial.println(count);
    delay(10);
}
