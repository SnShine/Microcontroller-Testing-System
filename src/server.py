'''
server
======

Structure of DATA from IP task:


----------------------------------------

'''

import socket
import sys


def send_data(data):
    '''gets data from IP task'''
    global DATA
    DATA= data
    parse_data(DATA)


def start_server():
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the port
    server_address = ('192.168.0.103', 8606)
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
    global FPS

    NAME, STATUS, COLOR_NAME, COLOR_RGB, FREQUENCY, FPS= DATA

def parse_command(command):
    a= command.split(":")
    if len(a)== 2 or len(a)== 3:
        return a
    else:
        return None

def LED_with_3(values):
    '''command about LED properties with three parameters'''
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
        if FREQUENCY[NAME.index(values[1])]== -1:
            return "The requensted LED hasn't switched on yet!"
        elif FREQUENCY[NAME.index(values[1])]== -2:
            return "The requested LED hasn't switched off yet!"
        else:
            return "{0:.2f}".format(FREQUENCY[NAME.index(values[1])])+ " Hz"

def LED_with_2(values):
    '''command about LED properties with two parameters'''
    if values[1]== "numbof":
        return str(len(NAME))+" LEDs"
    else:
        return ", ".join(NAME)

def IMAGE_with_3(values):
    '''command about IMAGE properties with three parameters'''
    return 0

def IMAGE_with_2(values):
    '''command about IMAGE properties with two parameters'''
    if values[1]== "fps":
        return str(FPS)+ " fps"

def talk_to_client(sock, connection):
    try:
        command = connection.recv(128)

        if len(command)!= 0:
            print
            print "Received command:", command

            # if 'close all' is received stop the server and exit
            if command== "close all":
                print "Closing connection and stopping server..."
                connection.close()
                sock.close()
                return 0

            else:
                values= parse_command(command)
                # if not properly formatted, return 'invalid command' to client
                if values== None:
                    print("Invalid command! Please check your command.")
                    connection.sendall("Invalid command! Please check your command.")
                else:
                    # if the command is about LED properties
                    if values[0]== "LED":
                        if len(values)== 3:
                            if values[1] not in NAME:
                                print("Please check the name of LED you have entered.")
                                connection.sendall("Please check the name of LED you have entered.")
                            elif values[2] not in ["status", "color_name", "color_rgb", "frequency"]:
                                print("Please check the property of LED you have entered.")
                                connection.sendall("Please check the property of LED you have entered.")
                            else:
                                answer= LED_with_3(values)
                                if answer== None:
                                    answer= "Haven't calculated the requested answer yet!"
                                print "Response to client:", answer
                                try:
                                    connection.sendall(answer)
                                except Exception, e:
                                    print e
                                    print "Error: ", sys.exc_info()[0]
                                    connection.sendall("Error sending the answer from server to client!")

                        elif len(values)== 2:
                            if values[1] not in ["numbof", "list"]:
                                print("Please check the property of LED_LIST you have entered.")
                                connection.sendall("Please check the property of LED_LIST you have entered.")
                            else:
                                answer= LED_with_2(values)
                                print "Response to client:", answer
                                try:
                                    connection.sendall(answer)
                                except Exception, e:
                                    print e
                                    print "Error: ", sys.exc_info()[0]
                                    connection.sendall("Error sending the answer from server to client!")
                        else:
                            print("Invalid number of LED specifications.")
                            connection.sendall("Invalid number of LED specifications.")

                    # if the command is about IMAGE properties
                    elif values[0]== "IMAGE":
                        if len(values)== 3:
                            pass
                        elif len(values)== 2:
                            if values[1] not in ["fps"]:
                                print("Please check the property of IMAGE you have entered.")
                                connection.sendall("Please check the property of IMAGE you have entered.")
                            else:
                                answer= IMAGE_with_2(values)
                                print "Response to client:", answer
                                try:
                                    connection.sendall(answer)
                                except Exception, e:
                                    print e
                                    print "Error: ", sys.exc_info()[0]
                                    connection.sendall("Error sending the answer from server to client!")
                        else:
                            print("Invalid number of IMAGE specifications.")
                            connection.sendall("Invalid number of IMAGE specifications.")

                    # ifcommand doesn't belong to any of these categories
                    else:
                        print("Please check the Element_Type you have entered.")
                        connection.sendall("Please check the Element_Type you have entered.")


        else:
            # if client got disconnected from server
            # press CTRL+c in client to disconnect!
            print "Client disconnected the connection. Stopping the server..."
            connection.close()
            sock.close()
            return 0
    except Exception, e:
        print e
        print "Error: ", sys.exc_info()[0]
        print "Error with server!"
        sys.exit()


def run():
    print __doc__
    sock, connection= start_server()

    while True:
        ret= talk_to_client(sock, connection)
        if ret== 0:
            break

#main()

