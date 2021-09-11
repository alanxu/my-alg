import socket
from asyncio import IncompleteReadError  # only import the exception class


class SocketStreamReader:
    def __init__(self, sock: socket.socket):
        self._sock = sock
        self._recv_buffer = bytearray()

    def read(self, num_bytes: int = -1) -> bytes:
        raise NotImplementedError

    def readexactly(self, num_bytes: int) -> bytes:
        buf = bytearray(num_bytes)
        pos = 0
        while pos < num_bytes:
            n = self._recv_into(memoryview(buf)[pos:])
            if n == 0:
                raise IncompleteReadError(bytes(buf[:pos]), num_bytes)
            pos += n
        return bytes(buf)

    def readline(self) -> bytes:
        return self.readuntil(b"\n")

    def readuntil(self, separator: bytes = b"\n") -> bytes:
        if len(separator) != 1:
            raise ValueError("Only separators of length 1 are supported.")

        chunk = bytearray(4096)
        start = 0
        buf = bytearray(len(self._recv_buffer))
        bytes_read = self._recv_into(memoryview(buf))
        assert bytes_read == len(buf)

        while True:
            idx = buf.find(separator, start)
            if idx != -1:
                break

            start = len(self._recv_buffer)
            bytes_read = self._recv_into(memoryview(chunk))
            buf += memoryview(chunk)[:bytes_read]

        result = bytes(buf[: idx + 1])
        self._recv_buffer = b"".join(
            (memoryview(buf)[idx + 1:], self._recv_buffer)
        )
        return result

    def _recv_into(self, view: memoryview) -> int:
        bytes_read = min(len(view), len(self._recv_buffer))
        view[:bytes_read] = self._recv_buffer[:bytes_read]
        self._recv_buffer = self._recv_buffer[bytes_read:]
        if bytes_read == len(view):
            return bytes_read
        bytes_read += self._sock.recv_into(view[bytes_read:])
        return bytes_read


import socket
import sys


class Cache:
    def __init__(self):
        self._cache = {'a': 'abc', 'b': 'bcd'}
        # Create a TCP/IP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn = None
        self.reader = None

    def get(self, key):
        return self._cache.get(key)

    def set(self, key, value):
        self._cache[key] = value

    def start(self, port=10000):
        # Bind the socket to the port
        server_address = ('localhost', port)
        self.sock.bind(server_address)

        self.sock.listen(1)
        print(f"Server started, waiting for connection. ")
        while True:
            # Wait for a connection
            connection, client_address = self.sock.accept()
            self.conn = connection
            self.reader = SocketStreamReader(self.sock)
            break

        print("Connection established!")

        try:
            # Receive the data in small chunks and retransmit it
            while True:
                data = self.conn.recv(16)
                print(data)
                data =
                # cmd = self.reader.readline()
                cmd = cmd.split(' ')
                print(f"Command received {cmd}")
                ans = []
                if cmd[0].lower() == 'get':

                    for key in cmd[1:]:
                        ans.append(self._cache[key])
                else:
                    pass

                for data in ans:
                    self.sock.send(data.encode())

        finally:
            # Clean up the connection
            connection.close()


cache = Cache()
cache.start(10001)