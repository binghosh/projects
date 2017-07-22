#!/usr/bin/env python

#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Mark T. Smith 2016
#

# This reads the details in a tag-it compliant tag.  It will reveal the
# following data items in the tag:
#
# Transponder ID
# Manufacturer Number
# Version Number
# Number of blocks
# Number of bytes per block
#
# See TI 6350 user manual for more information.
#
#
import io
import sys
import serial
import time


#
# Check that there is at least one argument which hopefully will be
# the serial port ID that is to be used.
#

#if len(sys.argv) < 2 :
#    print ("Usage: " + sys.argv[0] + " serial_port_to_use")
#    sys.exit()

#
# The TI reader defaults to 57600 baud, 8 bit data, 1 stop bit and no parity.
# There is no handshaking.
#
# Note that the timeout here is set to 5 seconds.  That is more than enough
# time to allow the TI RFID reader to turn on its radio, command a tag, and get
# data back from it.  We assume that if we time out and we don't have any data
# then the RFID reader is not on line.
#

var = 1


try:
	tiser = serial.Serial('COM5', baudrate=57600, bytesize=8,
		parity='N', stopbits=1, timeout=5, xonxoff=0, rtscts=0, dsrdtr=0)
except:
	print ("Usage: " + sys.argv[0] + " serial_port_to_use")
	print ("Can't open " + sys.argv[1] + ".")
	print ("Under linux or Apple OS you need the full path, ie /dev/ttyUSB0.")
	print ("Under windows use the communication port name, ie COM1")
	#sys.exit()
while var == 1 : 
	#
	# Form a read transponder details command.  To use the write() method in python
	# it needs to be in the form of a string or a buffer, which is just a pointer
	# into memory.  This code forms an array of bytes from a list that contains
	# the command to send and then uses a buffer (memoryview) to write it out.
	#

	read_transponder_details = [0x01, 0x09, 0, 0, 0, 0, 0x05, 0x0d, 0xf2]
	command = bytearray(9)
	idx = 0

	for i in read_transponder_details:
		command[idx] = i
		idx += 1

	#x_str = raw_input ("Enter any string to get TAGIT transponder details: ")
	tiser.write(memoryview(command))  # memoryview is the same as buffer

	#
	# We read the returned data from the reader in 2 passes.  First we read
	# the first two bytes.  The second byte is the length of the entire returned
	# packet.  From that we determine how many more bytes to read which are then
	# read in the second pass.
	#

	line_size = tiser.read(2)  # first pass, read first two bytes of reply

	if len(line_size) < 2:
		print ("No data returned.  Is the reader turned on?")
		#tiser.close()
		#sys.exit()

	# second pass
	#    print ("Reply length is " + str(ord(line_size[1])) + " bytes.")
	line_data = tiser.read((ord(line_size[1]) - 2))  # get the rest of the reply
	#    print ("I read " + str(len(line_data)) + " bytes.")

	#
	# The returned data is in the form of string objects.  Use that data to form
	# a single response list of integers.  Integers are exactly what the RFID reader
	# is sending back.  Doing this makes it easier to process the returned data.
	#

	response_len = ord(line_size[1]) # this is the length of the entire response
	response = []
	idx = 0

	response.append(ord(line_size[0])) # response SOF
	response.append(ord(line_size[1])) # response size
	# In the next line the -2 accounts for the SOF and size bytes done above.
	while idx < (response_len - 2): # do the rest of the response
		response.append(ord(line_data[idx]))
		idx += 1

	#
	# Compute the checksum.  To compute the checksum of the returned data you just
	# take the XOR of all the data bytes that were returned and compare with the checksum
	# bytes that were returned.
	#
	# The 'while' statment ranges from 0 to the length of the returned data - 2.  The
	# minus 2 is to adjust for the index (we number from 0) and also so that we do not
	# include the returned last checksum bytes in our own calculation.  We compute the
	# checksum on the returned data bytes, but not including the returned checksum bytes.
	#

	chksum = 0
	idx = 0
	while idx < (response_len - 2):
		chksum ^= response[idx]
		idx += 1

	#
	# Compare the checksums and if they don't match then bail out.
	# If they do match then all is well and all that remains is to
	# dig out and print the tag data.
	#

	if chksum != (response[response_len - 2]):  # and compare them
		print("Checksum error!")
	#    print (chksum)
	#    print (response[response_len - 2])
		#tiser.close()
		#sys.exit()

	#
	# See if the reply says that some sort of error has occurred.  This
	# is done by checking bit 4 in the the command flags byte.  If it is
	# set then we have an error.  The code for the error will be in the
	# data field of the reply.  A dictionary is then used to look up what
	# it means.
	#

	if (response[5] & 0x10) != 0:

		error_meaning = {
			"0x1" : "Transponder not found.",
			"0x2" : "Command not supported.",
			"0x3" : "Packet checksum invalid.",
			"0x4" : "Packet flags invalid for command.",
			"0x5" : "General write failure.",
			"0x6" : "Write failure due to locked block.",
			"0x7" : "Transponder does not support function.",
			"0xf" : "Undefined error."
			}.get(hex(response[7]), "Unknown error code.")

		print("Reader returned error code: " + hex(response[7]))
		print error_meaning
		#tiser.close()
		#sys.exit()

	#
	# If we get here, everything worked and we have data to print out.
	#

	print("Transponder ID: " + hex(response[10]) + hex(response[9])[2:4]
		+ hex(response[8])[2:4] + hex(response[7])[2:4])

	print("Manufacturer Number: " + hex(response[11]))

	print("Version Number: " + hex(response[13])
		+ hex(response[12])[2:4]) # the [2:4] cuts off the 0x

	print("Number of Blocks: " + hex(response[14]))

	print("Bytes per Block: " + hex(response[15]))
	time.sleep( 5 )

	#tiser.close()
