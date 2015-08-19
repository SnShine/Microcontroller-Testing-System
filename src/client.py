import socket
import sys

def start_client():
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect the socket to the port where the server is listening
    server_address = ('172.16.73.218', 8604)
    print "Connecting to", server_address, "..."

    try:
        sock.connect(server_address)
        print "Sucessfully connected to server!"
    except:
        print "Unable to connect to the server!"
        sys.exit()

    return sock

def talk_to_server(sock):
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
        data= sock.recv(64)
        print "Response: ", data
    except:
        print "Error has occured! (client)"
        print(sys.stderr)
        sock.close()
        sys.exit()


if __name__== "__main__":
    sock= start_client()

    while True:
        ret= talk_to_server(sock)
        if ret== 0:
            break
        