import socket
import sys

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect the socket to the port where the server is listening
server_address = ('172.16.48.241', 8680)
print ("Connecting to", server_address) 
sock.connect(server_address)
print ("Sucessfully connected to server...")

while True:
    print ("\nEnter your command: ",)
    user_command= raw_input()
    print(user_command)
    try:
        # Send data
        message = user_command
        #print ("Sending command to server...")
        sock.sendall(message)
        print("ppp")
        if message== "close all":
            print("Closing connection and stopping server!")
            sock.close()
            break
        # receive data
        data= sock.recv(64)
        print ("Response: ", data)
    except:
        print("Error has occured! (client)")
        print(sys.stderr)
        pass