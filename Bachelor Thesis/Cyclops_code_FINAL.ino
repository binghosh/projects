int LED1 = 2;
int LED2 = 3;
int LED3 = 4;
int LED4 = 5;
int LED5 = 6;
int LED6 = 7;

char incomingByte ;





void setup (){
  Serial.begin(9600);
  pinMode (LED1, OUTPUT);
  pinMode (LED2, OUTPUT);
  pinMode (LED3, OUTPUT);
  pinMode (LED4, OUTPUT);
  pinMode (LED5, OUTPUT);
  pinMode (LED6, OUTPUT);
 
  
  Serial.println("Start") ;
  
  //delay(500);
  
}

void loop () {
  
  if (Serial.available() > 0) {
    incomingByte = (char)Serial.read();
    
    if ((incomingByte == 'a' ) or (incomingByte == 'A' )){
      digitalWrite (LED1, HIGH);
     digitalWrite (LED2, LOW);
     digitalWrite (LED3, LOW);
     digitalWrite (LED4, LOW);
     digitalWrite (LED5, LOW);
     digitalWrite (LED6, LOW);
     Serial.print("a");
      
    }
    else if ((incomingByte == 'b' ) or (incomingByte == 'B' )){
     digitalWrite (LED1, HIGH);
      digitalWrite (LED2, HIGH);
      digitalWrite (LED3, LOW);
      digitalWrite (LED4, LOW);
      digitalWrite (LED5, LOW);
      digitalWrite (LED6, LOW); 
    Serial.print("b");
  } 
  else if ((incomingByte == 'c' ) or (incomingByte == 'C' )){
     digitalWrite (LED1, HIGH);
      digitalWrite (LED2, LOW);
      digitalWrite (LED3, LOW);
      digitalWrite (LED4, HIGH);
      digitalWrite (LED5, LOW);
      digitalWrite (LED6, LOW); 
    Serial.print("c");
  } 
  else if ((incomingByte == 'd' ) or (incomingByte == 'D' )){
     digitalWrite (LED1, HIGH);
      digitalWrite (LED2, LOW);
      digitalWrite (LED3, LOW);
      digitalWrite (LED4, HIGH);
      digitalWrite (LED5, HIGH);
      digitalWrite (LED6, LOW);
     Serial.print("d"); 
  }
  else if ((incomingByte == 'e' ) or (incomingByte == 'E' )){
     digitalWrite (LED1, HIGH);
      digitalWrite (LED2, LOW);
      digitalWrite (LED3, LOW);
      digitalWrite (LED4, LOW);
      digitalWrite (LED5, HIGH);
      digitalWrite (LED6, LOW);
     Serial.print("e"); 
  }
  else if ((incomingByte == 'f' ) or (incomingByte == 'F' )){
     digitalWrite (LED1, HIGH);
      digitalWrite (LED2, HIGH);
      digitalWrite (LED3, LOW);
      digitalWrite (LED4, HIGH);
      digitalWrite (LED5, LOW);
      digitalWrite (LED6, LOW);
     Serial.print("f"); 
  }
  else if ((incomingByte == 'g' ) or (incomingByte == 'G' )){
     digitalWrite (LED1, HIGH);
      digitalWrite (LED2, HIGH);
      digitalWrite (LED3, LOW);
      digitalWrite (LED4, HIGH);
      digitalWrite (LED5, HIGH);
      digitalWrite (LED6, LOW);
     Serial.print("g"); 
  }
  else if ((incomingByte == 'h' ) or (incomingByte == 'H' )){
     digitalWrite (LED1, HIGH);
      digitalWrite (LED2, HIGH);
      digitalWrite (LED3, LOW);
     digitalWrite (LED4, LOW);
      digitalWrite (LED5, HIGH);
      digitalWrite (LED6, LOW);
     Serial.print("h"); 
  }
  else if ((incomingByte == 'i' ) or (incomingByte == 'I' )){
     digitalWrite (LED1, LOW);
      digitalWrite (LED2, HIGH);
      digitalWrite (LED3, LOW);
      digitalWrite (LED4, HIGH);
      digitalWrite (LED5, LOW);
      digitalWrite (LED6, LOW); 
      Serial.print("i");
  }
  else if ((incomingByte == 'j' ) or (incomingByte == 'J' )){
     digitalWrite (LED1, LOW);
      digitalWrite (LED2, HIGH);
      digitalWrite (LED3, LOW);
      digitalWrite (LED4, HIGH);
      digitalWrite (LED5, HIGH);
     digitalWrite (LED6, LOW); 
     Serial.print("j");
  }
  else if ((incomingByte == 'k' ) or (incomingByte == 'K' )){
     digitalWrite (LED1, HIGH);
      digitalWrite (LED2, LOW);
      digitalWrite (LED3, HIGH);
      digitalWrite (LED4, LOW);
      digitalWrite (LED5, LOW);
      digitalWrite (LED6, LOW); 
      Serial.print("k");
  }
  else if ((incomingByte == 'l' ) or (incomingByte == 'L' )){
     digitalWrite (LED1, HIGH);
      digitalWrite (LED2, HIGH);
      digitalWrite (LED3, HIGH);
      digitalWrite (LED4, LOW);
      digitalWrite (LED5, LOW);
      digitalWrite (LED6, LOW);
     Serial.print("l"); 
    
  }
  else if ((incomingByte == 'm' ) or (incomingByte == 'M' )){
     digitalWrite (LED1, HIGH);
      digitalWrite (LED2, LOW);
      digitalWrite (LED3, HIGH);
      digitalWrite (LED4, HIGH);
      digitalWrite (LED5, LOW);
      digitalWrite (LED6, LOW); 
      Serial.print("m");
  }
  else if ((incomingByte == 'n' ) or (incomingByte == 'N' )){
     digitalWrite (LED1, HIGH);
      digitalWrite (LED2, LOW);
      digitalWrite (LED3, HIGH);
      digitalWrite (LED4, HIGH);
      digitalWrite (LED5, HIGH);
     digitalWrite (LED6, LOW); 
     Serial.print("n");
  }
  else if ((incomingByte == 'o' ) or (incomingByte == 'O' )){
     digitalWrite (LED1, HIGH);
      digitalWrite (LED2, LOW);
      digitalWrite (LED3, HIGH);
      digitalWrite (LED4, LOW);
      digitalWrite (LED5, HIGH);
      digitalWrite (LED6, LOW); 
      Serial.print("o");
  }
  else if ((incomingByte == 'p' ) or (incomingByte == 'P' )){
     digitalWrite (LED1, HIGH);
      digitalWrite (LED2, HIGH);
      digitalWrite (LED3, HIGH);
      digitalWrite (LED4, HIGH);
      digitalWrite (LED5, LOW);
      digitalWrite (LED6, LOW); 
      Serial.print("p");
  }
  else if ((incomingByte == 'q' ) or (incomingByte == 'Q' )){
     digitalWrite (LED1, HIGH);
      digitalWrite (LED2, HIGH);
      digitalWrite (LED3, HIGH);
      digitalWrite (LED4, HIGH);
      digitalWrite (LED5, HIGH);
      digitalWrite (LED6, LOW);
     Serial.print("q"); 
  }
  else if ((incomingByte == 'r' ) or (incomingByte == 'R' )){
     digitalWrite (LED1, HIGH);
      digitalWrite (LED2, HIGH);
      digitalWrite (LED3, HIGH);
      digitalWrite (LED4, LOW);
      digitalWrite (LED5, HIGH);
      digitalWrite (LED6, LOW); 
      Serial.print("r");
  }
  else if ((incomingByte == 's' ) or (incomingByte == 'S' )){
     digitalWrite (LED1, LOW);
      digitalWrite (LED2, HIGH);
      digitalWrite (LED3, HIGH);
      digitalWrite (LED4, HIGH);
      digitalWrite (LED5, LOW);
      digitalWrite (LED6, LOW);
     Serial.print("s"); 
  }
  else if ((incomingByte == 't' ) or (incomingByte == 'T' )){
     digitalWrite (LED1, LOW);
      digitalWrite (LED2, HIGH);
      digitalWrite (LED3, HIGH);
      digitalWrite (LED4, HIGH);
      digitalWrite (LED5, HIGH);
      digitalWrite (LED6, LOW); 
      Serial.print("t");
  }
  else if ((incomingByte == 'u' ) or (incomingByte == 'U' )){
     digitalWrite (LED1, HIGH);
      digitalWrite (LED2, LOW);
      digitalWrite (LED3, HIGH);
      digitalWrite (LED4, LOW);
      digitalWrite (LED5, LOW);
      digitalWrite (LED6, HIGH); 
      Serial.print("u");
  }
  else if ((incomingByte == 'v' ) or (incomingByte == 'V' )){
     digitalWrite (LED1, HIGH);
      digitalWrite (LED2, HIGH);
      digitalWrite (LED3, HIGH);
      digitalWrite (LED4, LOW);
      digitalWrite (LED5, LOW);
      digitalWrite (LED6, HIGH);
     Serial.print("v"); 
  }
  else if ((incomingByte == 'w' ) or (incomingByte == 'W' )){
     digitalWrite (LED1, HIGH);
      digitalWrite (LED2, LOW);
      digitalWrite (LED3, HIGH);
      digitalWrite (LED4, HIGH);
      digitalWrite (LED5, LOW);
      digitalWrite (LED6, HIGH); 
      Serial.print("w");
  }
  else if ((incomingByte == 'x' ) or (incomingByte == 'X' )){
     digitalWrite (LED1, HIGH);
      digitalWrite (LED2, LOW);
      digitalWrite (LED3, HIGH);
      digitalWrite (LED4, HIGH);
      digitalWrite (LED5, HIGH);
      digitalWrite (LED6, HIGH); 
      Serial.print("x");
  }
  else if ((incomingByte == 'y' ) or (incomingByte == 'Y' )){
     digitalWrite (LED1, HIGH);
      digitalWrite (LED2, LOW);
      digitalWrite (LED3, HIGH);
      digitalWrite (LED4, LOW);
      digitalWrite (LED5, HIGH);
      digitalWrite (LED6, HIGH); 
      Serial.print("y");
  }
  else if ((incomingByte == 'z' ) or (incomingByte == 'Z' )){
     digitalWrite (LED1, LOW);
      digitalWrite (LED2, HIGH);
      digitalWrite (LED3, HIGH);
      digitalWrite (LED4, HIGH);
      digitalWrite (LED5, HIGH);
      digitalWrite (LED6, HIGH); 
      Serial.print("z");
  }
  else if ((incomingByte == ' ' ) ){
     digitalWrite (LED1, HIGH);
      digitalWrite (LED2, HIGH);
      digitalWrite (LED3, HIGH);
      digitalWrite (LED4, HIGH);
      digitalWrite (LED5, HIGH);
      digitalWrite (LED6, HIGH); 
      Serial.print("z");
  
  }
   else if ((incomingByte == 'abc ' ) ){
     digitalWrite (LED1, LOW);
      digitalWrite (LED2, LOW);
      digitalWrite (LED3, LOW);
      digitalWrite (LED4, LOW);
      digitalWrite (LED5, LOW);
      digitalWrite (LED6, LOW); 
      Serial.print("z");
  } 
  }  
}
