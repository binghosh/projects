import serial
ser = serial.Serial('/dev/ttyACM0')  # open serial port
print(ser.name)         # check which port was really used
print ser.read(70)     # write a string
ser.close()  
