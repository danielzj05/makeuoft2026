
const int GSR = A0;
int sensorVal = 0;
int gsr_average = 0;



void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
}

void loop() {
  long sum = 0;
  for (int i = 0; i<10;i++)
    {
      sensorVal = analogRead(GSR);
      sum += sensorVal;
      delay(5);
    }
  gsr_average = sum/10;
  Serial.print("gsr_average =");
  Serial.print(gsr_average);
  int human_resistance = ((1024+2*gsr_average)*10000)/(516-gsr_average);
  Serial.print("    Human_resistance = ");
  Serial.println(human_resistance);
}







