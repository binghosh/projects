from tkinter import Tk
from time import sleep
r = Tk()
r.withdraw()
while not r.selection_get(selection = "CLIPBOARD"):
    sleep(0.1)
result = r.selection_get(selection = "CLIPBOARD")
r.clipboard_clear()
#r.clipboard_append('This goes to clipboard')
r.destroy()
print(result)

import serial
import time

arduino = serial.Serial('COM8', 9600, timeout=.1)
time.sleep(2) # waiting the initialization...
print("initialising")
#data = "HLHLHLHLHLHLHLHLHLHLHLHLHLHLHLHLHLHLHHL"
for char in result:
    #print(char, end='')
    time.sleep(5)
    arduino.write(bytes(char, 'UTF-8')) # turns LED ON
#print("LED ON")
time.sleep(2) # waits for 2 second

#arduino.write(b"L") # turns LED OFF
#print("LED OFF")
#time.sleep(60) # waits for 1 s


while 1:
    
    sensorValue = arduino.read()
    print(sensorValue)
arduino.close() #say goodbye to Arduino
