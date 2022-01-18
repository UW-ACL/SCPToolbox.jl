import numpy as np
import numpy.linalg as la
import socket
import importlib
import traceback

import os
import sys
sys.path.append("../../")

class Server:
    def __init__(self,ip,port):
        self.msg_len = 1024 # [B] Message size
        self.sz_len = 4 # [B] Length of the "message length" content
        self.max_connection_attempts = 10

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.port = self.connect(ip,port)
        if self.port==-1:
            raise RuntimeError("[Server] Failed to establish connection\n")

    def __del__(self):
        self.s.close()

    def connect(self,ip,port):
        """
        Establish network connection.

        Parameters
        ----------
        ip : str
            IP address.
        port : int
            Port.

        Returns
        -------
        port : int
            Which port connected to. If -1, means connection failed.
        """
        attempt_count = self.max_connection_attempts
        while attempt_count > 0:
            try:
                self.s.bind((ip,port))
                self.s.listen(True)
                return port
            except:
                print("[Server]: Port %d already in use. Binding to port: %d"
                      %(port,port+1))
                port += 1
                attempt_count -= 1
        return -1

    def wait_connection(self):
        """
        Blocking wait for connection from client.

        Returns
        -------
        conn : Socket
            Object usable to send and receive data on the connection.
        """
        print("[Server]: Waiting for connection...")
        conn,addr = self.s.accept()
        print("[Server]: Connected")
        return conn

    def receive(self,conn):
        """
        Receive data on connection. Protocol: first self.sz_len bytes of the
        buffer are the length of the string in the following message.

        Returns
        -------
        msg : str
            The received message content.
        """
        buf_length = self.msg_len
        buf = b''
        while buf_length:
            newbuf = conn.recv(self.msg_len)
            buf += newbuf
            buf_length -= len(newbuf)
        msg_length = int(buf[:self.sz_len])
        msg = buf[self.sz_len:self.sz_len+msg_length]
        print("[Server] Received %s"%(msg))
        return msg

    def send(self,message,conn):
        """
        Send data
        """
        msg_sz = str(len(message)).ljust(self.sz_len)
        msg = msg_sz+message
        print("[Server] Sending %s"%(msg))
        conn.sendall(msg.encode())  # Send message

if __name__ == "__main__":
    server = Server(ip="127.0.0.1",port=1234)

    while True:
        # ..:: Receive message for processing ::..
        conn = server.wait_connection()
        msg = server.receive(conn)
        # ..:: Import custom functions that may have been defined ::..
        try:
            import pykz_custom
            importlib.reload(pykz_custom)
            from pykz_custom import *
        except Exception as e:
            print("[Server] Unable to import pykz_custom ("+str(e)+")")
        msg = msg.decode('ascii')
        try:
            if "<function>" in msg:
                # ..:: Process function ::..
                method_name = msg[11:-1]
                exec('output=' + method_name)
                # output = getattr(pykz_custom, method_name)()
                if type(output) is dict:
                    for key,val in output.items():
                        exec(key + '=val') # Save outputs
                msg = "executed %s"%(method_name)
                server.send(str(msg),conn)
            elif "<exec>" in msg:
                # ..:: Process custom one-line "any" command ::..
                cmd = msg[7:]
                exec(cmd) # Execute command
                msg = "executed"
                server.send(str(msg),conn)
            else:
                # ..:: Process one-liner result=... command ::..
                result=eval(msg) # Evaluate message
                server.send(str(result),conn)
        except:
            traceback.print_exc()
            server.send('PYKZ_FAIL',conn)
