import Tkinter as tk
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import io
import sys
import serial


# Set number of rows and columns
ROWS = 4
COLS = 4
S1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
S2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
S3 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# Create a grid of None to store the references to the tiles
tiles = [[None for _ in range(COLS)] for _ in range(ROWS)]

def callback(event):
    # Get rectangle diameters
    col_width = c.winfo_width()/COLS
    row_height = c.winfo_height()/ROWS
    # Calculate column and row number
    col = event.x//col_width
    row = event.y//row_height
    # If the tile is not filled, create a rectangle
    if not tiles[row][col]:
        tiles[row][col] = c.create_rectangle(col*col_width, row*row_height, (col+1)*col_width, (row+1)*row_height, fill="Green")
	#try:
	tiser = serial.Serial('/dev/ttyUSB0', baudrate=57600, bytesize=8,
		parity='N', stopbits=1, timeout=0.5, xonxoff=0, rtscts=0, dsrdtr=0)
	print "done"
	carrier_onoff_command = [0x01, 0x0a, 0, 0, 0, 0, 0xf4, 0, 0, 0]

	command = bytearray(10)
	idx = 0

	for i in carrier_onoff_command:
	    command[idx] = i
	    idx += 1

	arg = "on "  # init this to be some string

	

	if (arg == 'on') or (arg == 'ON'):

	    command[7] = 0xff

	command_len = len(carrier_onoff_command)
	chksum = 0
	idx = 0
	while idx < (command_len - 2):
	    chksum ^= command[idx]
	    idx += 1

	command[command_len - 2] = chksum  # 1st byte is the checksum
	command[command_len - 1] = chksum ^ 0xff  # 2nd byte is ones comp of checksum

	tiser.write(memoryview(command))  # memoryview == buffer


	line_size = tiser.read(2)  # first pass, read first two bytes of reply

	if len(line_size) < 2:
	    print ("No data returned.  Is the reader turned on?")
	    sys.exit()

	line_data = tiser.read((ord(line_size[1]) - 2))  # get the rest of the reply

	response_len = ord(line_size[1]) # this is the length of the entire response
	response = []
	idx = 0

	response.append(ord(line_size[0])) # response SOF
	response.append(ord(line_size[1])) # response size

	while idx < (response_len - 2): # do the rest of the response
	    response.append(ord(line_data[idx]))
	    idx += 1
	chksum = 0
	idx = 0
	while idx < (response_len - 2):
	    chksum ^= response[idx]
	    idx += 1

	if chksum == (response[response_len - 2]):  # and compare them

	    if response[7] == 0:
	        print("Carrier successfully turned " + arg + ".")
	    else:
	        print("Command execution error, returned code is " + response[7] + ".")
	        print("Carrier state not changed.")

	else:
	    print("Checksum error!")
	tiser.close()
	
	ser = serial.Serial('/dev/ttyACM0')  # open serial port
	print(ser.name)         # check which port was really used
	ser.write(b's')
	line = ser.readline()
	#print ser.readline()     # write a string
	print line
	test =line.split(",")
   	test =  [float(S) for S in test]
	if (row==0 and col==0) :
		INDEX =0
	elif (row==0 and col==1) :
		INDEX =1
	elif (row==0 and col==2) :
		INDEX =2
	elif (row==0 and col==3) :
		INDEX =3
	elif (row==1 and col==0) :
		INDEX =4
	elif (row==1 and col==1) :
		INDEX =5
	elif (row==1 and col==2) :
		INDEX =6
	elif (row==1 and col==3) :
		INDEX =7
	elif (row==2 and col==0) :
		INDEX =8
	elif (row==2 and col==1) :
		INDEX =9
	elif (row==2 and col==2) :
		INDEX =10
	elif (row==2 and col==3) :
		INDEX =11
	elif (row==3 and col==0) :
		INDEX =12
	elif (row==3 and col==1) :
		INDEX =13
	elif (row==3 and col==2) :
		INDEX =14
	elif (row==3 and col==3) :
		INDEX =15
	S1[INDEX]= test[0]
	S2[INDEX]= test[1]
	S3[INDEX]= test[2]
	S1[INDEX+16]= test[3] 
	S2[INDEX+16]= test[4]
	S3[INDEX+16]= test[5]
	S1[INDEX+32]= test[6]
	S2[INDEX+32]= test[7]
	S3[INDEX+32]= test[8]
	S1[INDEX+48]= test[9]
	S2[INDEX+48]= test[10]
	S3[INDEX+48]= test[11]
	print S1
	print S2
	print S3
	print INDEX
	print row
	print col
	ser.close() 
	
    else:
        c.delete(tiles[row][col])
        tiles[row][col] = None



root = tk.Tk()



fields = 'Antenna-Width', 'Antenna-Breadth'

def fetch(entries):
   for entry in entries:
      field = entry[0]
      text  = entry[1].get()
      print('%s: "%s"' % (field, text))
   xtemp = entries[0][1].get()
   xtemp = float(xtemp)
   ytemp = entries[1][1].get()
   ytemp = float(ytemp)
   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')
   with open('numbers.txt') as f:
   	lines = f.read().splitlines()
   
   #x = lines[0].split(",")
   #x =  [float(S) for S in x] 
   x = [(xtemp*1)/4,(xtemp*2)/4,(xtemp*3)/4,(xtemp*4)/4,(xtemp*1)/4,(xtemp*2)/4,(xtemp*3)/4,(xtemp*4)/4,(xtemp*1)/4,(xtemp*2)/4,(xtemp*3)/4,(xtemp*4)/4,(xtemp*1)/4,(xtemp*2)/4,(xtemp*3)/4,(xtemp*4)/4,(xtemp*1)/4,(xtemp*2)/4,(xtemp*3)/4,(xtemp*4)/4,(xtemp*1)/4,(xtemp*2)/4,(xtemp*3)/4,(xtemp*4)/4,(xtemp*1)/4,(xtemp*2)/4,(xtemp*3)/4,(xtemp*4)/4,(xtemp*1)/4,(xtemp*2)/4,(xtemp*3)/4,(xtemp*4)/4,(xtemp*1)/4,(xtemp*2)/4,(xtemp*3)/4,(xtemp*4)/4,(xtemp*1)/4,(xtemp*2)/4,(xtemp*3)/4,(xtemp*4)/4,(xtemp*1)/4,(xtemp*2)/4,(xtemp*3)/4,(xtemp*4)/4,(xtemp*1)/4,(xtemp*2)/4,(xtemp*3)/4,(xtemp*4)/4,(xtemp*1)/4,(xtemp*2)/4,(xtemp*3)/4,(xtemp*4)/4,(xtemp*1)/4,(xtemp*2)/4,(xtemp*3)/4,(xtemp*4)/4,(xtemp*1)/4,(xtemp*2)/4,(xtemp*3)/4,(xtemp*4)/4,(xtemp*1)/4,(xtemp*2)/4,(xtemp*3)/4,(xtemp*4)/4]
   
   #y =lines[1].split(",")
   #y =  [float(S) for S in y] 
   y = [(ytemp*1)/4,(ytemp*1)/4,(ytemp*1)/4,(ytemp*1)/4,(ytemp*2)/4,(ytemp*2)/4,(ytemp*2)/4,(ytemp*2)/4,(ytemp*3)/4,(ytemp*3)/4,(ytemp*3)/4,(ytemp*3)/4,(ytemp*4)/4,(ytemp*4)/4,(ytemp*4)/4,(ytemp*4)/4,(ytemp*1)/4,(ytemp*1)/4,(ytemp*1)/4,(ytemp*1)/4,(ytemp*2)/4,(ytemp*2)/4,(ytemp*2)/4,(ytemp*2)/4,(ytemp*3)/4,(ytemp*3)/4,(ytemp*3)/4,(ytemp*3)/4,(ytemp*4)/4,(ytemp*4)/4,(ytemp*4)/4,(ytemp*4)/4,(ytemp*1)/4,(ytemp*1)/4,(ytemp*1)/4,(ytemp*1)/4,(ytemp*2)/4,(ytemp*2)/4,(ytemp*2)/4,(ytemp*2)/4,(ytemp*3)/4,(ytemp*3)/4,(ytemp*3)/4,(ytemp*3)/4,(ytemp*4)/4,(ytemp*4)/4,(ytemp*4)/4,(ytemp*4)/4,(ytemp*1)/4,(ytemp*1)/4,(ytemp*1)/4,(ytemp*1)/4,(ytemp*2)/4,(ytemp*2)/4,(ytemp*2)/4,(ytemp*2)/4,(ytemp*3)/4,(ytemp*3)/4,(ytemp*3)/4,(ytemp*3)/4,(ytemp*4)/4,(ytemp*4)/4,(ytemp*4)/4,(ytemp*4)/4]
   #Z1 =lines[2].split(",")
   #Z1 =  [float(S) for S in Z1] 
   Z1 =[.2,.2,.2,.2,.2,.2,.2,.2,.2,.2,.2,.2,.2,.2,.2,.2,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40]
   #A1 =lines[3].split(",")
   #A1 =  [float(S) for S in A1]
   #print S1
   #print S2
   #print S3
   #S11 = [S*5 for S in A1]
   #S12 = [S*4 for S in A1]
   #S13 = [S*3 for S in A1]
   x2 = [d+(xtemp*1)/40 for d in x]
   y2 = [d+(ytemp*1)/40 for d in y]
   Z2 = [d+0.1 for d in Z1]
   color=['c','m','g']
   sp=ax.scatter(x2, y, Z1, s=S1, c=color[0],depthshade=False, cmap=color[0],label="Strength in X-axis")
   sp=ax.scatter(x, y2, Z1, s=S2, c=color[1],depthshade=False, cmap=color[1],label="Strength in Y-axis")
   sp=ax.scatter(x, y, Z2, s=S3, c=color[2],depthshade=False, cmap=color[2],label="Strength in Z-axis")
   #plt.colorbar(sp)
   plt.legend();
   plt.show() 

def makeform(root, fields):
   entries = []
   for field in fields:
      row = tk.Frame(root)
      lab = tk.Label(row, width=15, text=field, anchor='w')
      ent = tk.Entry(row)
      row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
      lab.pack(side=tk.LEFT)
      ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
      entries.append((field, ent))
   return entries

if __name__ == '__main__':
   
   ents = makeform(root, fields)
   root.bind('<Return>', (lambda event, e=ents: fetch(e)))   
   b1 = tk.Button(root, text='Show',
          command=(lambda e=ents: fetch(e)))
   b1.pack(side=tk.LEFT, padx=5, pady=5)
   b2 = tk.Button(root, text='Quit', command=root.quit)
   b2.pack(side=tk.LEFT, padx=5, pady=5)

c1 =tk.Canvas(root, width=500, height=15, borderwidth=0, background='white')
c1.pack()
z=0.2
c1.create_text(250,10,text="Please click on the cordinates which you want to measure and move hight wise")
c = tk.Canvas(root, width=500, height=500, borderwidth=5, background='white')
c.pack()

c.bind("<Button-1>", callback)


root.mainloop()
