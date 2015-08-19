import socket
import sys

def start_server():
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the port
    server_address = ('172.16.73.218', 8608)
    print "Starting server on", server_address, "..."

    sock.bind(server_address)

    # Listen for incoming connections
    sock.listen(1)

    print "Waiting for connection..."
    connection, client_address = sock.accept()
    print "Connection from: ", client_address

    return sock, connection

def talk_to_client(sock, connection):
    try:
        data = connection.recv(64)
        print "Received data:", data

        if data:
            if data== "close all":
                print "Closing connection and stopping server..."
                connection.close()
                sock.close()
                return 0
                
            else:
                connection.sendall("Please enter a recognized command!")
        else:
            print "Client disconnected the connection. Stopping the server..."
            connection.close()
            sock.close()
            return 0         
    except:
        print "Error has occured! (server)"
        print sys.stderr
        sys.exit()


def run():
    sock, connection= start_server()

    while True:
        ret= talk_to_client(sock, connection)
        if ret== 0:
            break

#main()
        