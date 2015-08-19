import socket
import sys
'''
Structure of DATA from IP task:


'''

def send_data(data):
    '''gets data from IP task'''
    global DATA
    DATA= data
    parse_data(DATA)

 
def start_server():
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the port
    server_address = ('172.16.48.241', 8601)
    print "Starting server on", server_address, "..."

    sock.bind(server_address)

    # Listen for incoming connections
    sock.listen(1)

    print "Waiting for connection..."
    connection, client_address = sock.accept()
    print "Connection from: ", client_address

    return sock, connection

def parse_data(data):
    global NAME
    global STATUS
    global COLOR_NAME
    global COLOR_RGB
    global FREQUENCY
    
    NAME, STATUS, COLOR_NAME, COLOR_RGB, FREQUENCY= DATA

def parse_command(command):
    a= command.split(":")
    if len(a)== 3:
        return a
    else:
        return None
    
def find_answer(values):
    #print(values)
    if values[2]== "status":
        sta= STATUS[NAME.index(values[1])]
        if sta== True:
            return "On"
        else:
            return "Off"
    if values[2]== "color_name":
        return COLOR_NAME[NAME.index(values[1])]
    if values[2]== "color_rgb":
        rgb= COLOR_RGB[NAME.index(values[1])]
        return "R: "+ str(rgb[2])+ " G: "+ str(rgb[1])+ " B: "+ str(rgb[0])
    if values[2]== "frequency":
        return FREQUENCY[NAME.index(values[1])]


def talk_to_client(sock, connection):
    try:
        command = connection.recv(64)
        print
        print "Received command:", command

        if command:
            if command== "close all":
                print "Closing connection and stopping server..."
                connection.close()
                sock.close()
                return 0

            else:
                values= parse_command(command)
                if values== None:
                    print("Invalid command! Please check your command.")
                    connection.sendall("Invalid command! Please check your command.")
                else:
                    if values[0] not in ["LED"]:
                        print("Please check the Element_Type you have entered.")
                        connection.sendall("Please check the Element_Type you have entered.")
                    elif values[1] not in NAME:
                        print("Please check the Element_Name you have entered.")
                        connection.sendall("Please check the Element_Name you have entered.")
                    elif values[2] not in ["status", "color_name", "color_rgb", "frequency"]:
                        print("Please check the Element_Property you have entered.")
                        connection.sendall("Please check the Element_Property you have entered.")
                    else:
                        answer= find_answer(values)
                        #print(DATA)
                        #final answer to send
                        print "Response to client:", answer
                        try:
                            connection.sendall(answer)
                        except Exception, e:
                            print e
                            print "Error: ", sys.exc_info()[0]
                            connection.sendall("Error sending the answer from server to client!")
        else:
            print "Client disconnected the connection. Stopping the server..."
            connection.close()
            sock.close()
            return 0         
    except:
        print "Error has occured! (server)"
        print e
        print "Error: ", sys.exc_info()[0]
        sys.exit()


def run():
    sock, connection= start_server()

    while True:
        ret= talk_to_client(sock, connection)
        if ret== 0:
            break

#main()
        