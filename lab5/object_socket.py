import socket
import select
import pickle
import datetime

from typing import *


class ObjectSocketParams:
    """Class containing parameters and constants for ObjectSocket."""
    OBJECT_HEADER_SIZE_BYTES = 4
    DEFAULT_TIMEOUT_S = 1
    CHUNK_SIZE_BYTES = 1024

class ObjectSenderSocket:
    """Class for sending Python objects over a TCP socket."""
    ip: str
    port: int
    sock: socket.socket
    conn: socket.socket
    print_when_awaiting_receiver: bool
    print_when_sending_object: bool

    def __init__(self, ip: str, port: int,
                 print_when_awaiting_receiver: bool = False,
                 print_when_sending_object: bool = False):
        """
              Initialize ObjectSenderSocket.
              Parameters:
                  ip (str): IP address to bind the socket.
                  port (int): Port number to bind the socket.
                  print_when_awaiting_receiver (bool): Optional argument.Whether to print messages while awaiting receiver connection. Defaults to False.
                  print_when_sending_object (bool): Optional argument. Whether to print messages while sending objects. Defaults to False.
              """
        self.ip = ip
        self.port = port

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.ip, self.port))
        self.conn = None

        self.print_when_awaiting_receiver = print_when_awaiting_receiver
        self.print_when_sending_object = print_when_sending_object

        self.await_receiver_conection()

    def await_receiver_conection(self):
        """
               Await receiver connection.
               Waits for a receiver to connect to the sender socket and establishes a connection.
               Prints status messages based on the value of print_when_awaiting_receiver parameter.
               """

        if self.print_when_awaiting_receiver:
            print(f'[{datetime.datetime.now()}][ObjectSenderSocket/{self.ip}:{self.port}] awaiting receiver connection...')

        self.sock.listen(1)
        self.conn, _ = self.sock.accept()

        if self.print_when_awaiting_receiver:
            print(f'[{datetime.datetime.now()}][ObjectSenderSocket/{self.ip}:{self.port}] receiver connected')

    def close(self):
        """Close the socket connection."""
        self.conn.close()
        self.conn = None

    def is_connected(self) -> bool:
        """
                Check if the socket is connected.
                Returns:
                    bool: True if the socket is connected, False otherwise.
                """
        return self.conn is not None

    def send_object(self, obj: Any):
        """
                Send a Python object over the socket.
                Serializes the object using pickle, sends the object size followed by the serialized data.
                Parameters:
                    obj (Any): Python object to send.
                """
        data = pickle.dumps(obj)
        data_size = len(data)
        data_size_encoded = data_size.to_bytes(ObjectSocketParams.OBJECT_HEADER_SIZE_BYTES, 'little')
        self.conn.sendall(data_size_encoded)
        self.conn.sendall(data)
        if self.print_when_sending_object:
            print(f'[{datetime.datetime.now()}][ObjectSenderSocket/{self.ip}:{self.port}] Sent object of size {data_size} bytes.')



class ObjectReceiverSocket:
    """Class for receiving Python objects over a TCP socket."""

    ip: str
    port: int
    conn: socket.socket
    print_when_connecting_to_sender: bool
    print_when_receiving_object: bool

    def __init__(self, ip: str, port: int,
                 print_when_connecting_to_sender: bool = False,
                 print_when_receiving_object: bool = False):
        """
               Initialize ObjectReceiverSocket.
               Parameters:
                   ip (str): IP address of the sender.
                   port (int): Port number of the sender.
                   print_when_connecting_to_sender (bool): Optional variable, whether to print messages while connecting to the sender.
                   print_when_receiving_object (bool): Optional variable, whether to print messages while receiving objects.

               """
        self.ip = ip
        self.port = port
        self.print_when_connecting_to_sender = print_when_connecting_to_sender
        self.print_when_receiving_object = print_when_receiving_object

        self.connect_to_sender()

    def connect_to_sender(self):
        """
                Connect to the sender.
                Establishes a connection to the sender socket.
                Prints status messages based on the value of print_when_connecting_to_sender parameter.
                """

        if self.print_when_connecting_to_sender:
            print(f'[{datetime.datetime.now()}][ObjectReceiverSocket/{self.ip}:{self.port}] connecting to sender...')

        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn.connect((self.ip, self.port))

        if self.print_when_connecting_to_sender:
            print(f'[{datetime.datetime.now()}][ObjectReceiverSocket/{self.ip}:{self.port}] connected to sender')

    def close(self):
        """Close the socket connection."""
        self.conn.close()
        self.conn = None

    def is_connected(self) -> bool:
        """
               Check if the socket is connected.
               Returns:
                   bool: True if the socket is connected, False otherwise.
               """
        return self.conn is not None

    def recv_object(self) -> Any:
        """
                Receive a Python object from the sender.

                Receives the object size, followed by the serialized data, deserializes it using pickle, and returns the object.

                Returns:
                    Any: The received Python object.
                """
        obj_size_bytes = self._recv_object_size()
        data = self._recv_all(obj_size_bytes)
        obj = pickle.loads(data)
        if self.print_when_receiving_object:
            print(f'[{datetime.datetime.now()}][ObjectReceiverSocket/{self.ip}:{self.port}] Received object of size {obj_size_bytes} bytes.')
        return obj

    def _recv_with_timeout(self, n_bytes: int, timeout_s: float = ObjectSocketParams.DEFAULT_TIMEOUT_S) -> Optional[bytes]:
        """
                Receive data from the socket with a timeout.

                Parameters:
                    n_bytes (int): Number of bytes to receive.
                    timeout_s (float): Timeout value in seconds.

                Returns:
                    Optional[bytes]: Received data, or None if timeout occurs.
                """
        rlist, _1, _2 = select.select([self.conn], [], [], timeout_s)
        if rlist:
            data = self.conn.recv(n_bytes)
            return data
        else:
            return None  # Only returned on timeout

    def _recv_all(self, n_bytes: int, timeout_s: float = ObjectSocketParams.DEFAULT_TIMEOUT_S) -> bytes:
        """
                Receive all data from the socket with a timeout.

                Parameters:
                    n_bytes (int): Number of bytes to receive.
                    timeout_s (float): Timeout value in seconds.

                Returns:
                    bytes: Received data.
                """
        data = []
        left_to_recv = n_bytes
        while left_to_recv > 0:
            desired_chunk_size = min(ObjectSocketParams.CHUNK_SIZE_BYTES, left_to_recv)
            chunk = self._recv_with_timeout(desired_chunk_size, timeout_s)
            if chunk is not None:
                data += [chunk]
                left_to_recv -= len(chunk)
            else:  # no more data incoming, timeout
                bytes_received = sum(map(len, data))
                raise socket.error(f'Timeout elapsed without any new data being received. '
                                   f'{bytes_received} / {n_bytes} bytes received.')
        data = b''.join(data)
        return data

    def _recv_object_size(self) -> int:
        """
                Receive the size of the Python object.

                Returns:
                    int: Size of the object in bytes.
                """
        data = self._recv_all(ObjectSocketParams.OBJECT_HEADER_SIZE_BYTES)
        obj_size_bytes = int.from_bytes(data, 'little')
        return obj_size_bytes