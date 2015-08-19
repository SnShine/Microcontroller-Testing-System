import socket
import sys

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the port
server_address = ('172.16.48.241', 8680)
print "Starting server on", server_address

sock.bind(server_address)

# Listen for incoming connections
sock.listen(1)

print("Waiting for connection...")
connection, client_address = sock.accept()
print "Connection from: ", client_address

while True:
    try:
        data = connection.recv(64)
        print "Received data:", data

        if data== "close all":
            print("Closing connection and stopping server...")
            connection.close()
            sock.close()
            break
        else:
            connection.sendall("haha success!")
    except:
        print("Error has occured! (server)")
        pass