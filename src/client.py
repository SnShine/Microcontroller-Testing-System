'''
client
======

Recognized commands:
    close all - stops server and client. Press Esc to close the window

    LED:<name>:status - returns the status of the selected LED; On/Off
    LED:<name>:color_name - return color name of the selected LED; red/yellow/green/cyan/blue
    LED:<name>:color_rgb - returns color RGB values; R: 150 G: 60 B:190
    LED:<name>:freqyency - returns blinking frequency of the selected LED; 1.2 Hz

    LED:numbof - returns total number of LEDs; 5 LEDs
    LED:list - returns names of all the marked LEDs

    IMAGE:fps - returns frames per second of the current video; 27 fps

    IMAGE:store:int - 
    IMAGE:store:ext -

----------------------------------------

'''

import socket
import sys


def start_client():
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect the socket to the port where the server is listening
    server_address = ('172.16.48.241', 8600)
    print "Connecting to", server_address, "..."

    try:
        sock.connect(server_address)
        print "Sucessfully connected to server!"
    except:
        print "Unable to connect to the server!"
        sys.exit()

    return sock

def talk_to_server(sock):
    '''
    sends commands to server and displays response on the terminal
    '''
    print "\nEnter your command:",
    user_command= raw_input()
    #print(user_command)
    try:
        # Send command
        message = user_command
        sock.sendall(message)

        if message== "close all":
            print "Closing connection and stopping server!"
            sock.close()
            return 0
        
        # receive output
        response= sock.recv(128)
        print "Response from server:", response
    except:
        print "Error has occured! (client)"
        print(sys.stderr)
        sock.close()
        sys.exit()


def main():
    print __doc__
    
    sock= start_client()

    while True:
        ret= talk_to_server(sock)
        if ret== 0:
            break

main()
        