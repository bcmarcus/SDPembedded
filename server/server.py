import socket
import ssl
import time
import threading

class DataReceiver:
    def __init__(self, buffer_size=4096, timeout=10):
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.data_buffer = b""
        self.last_packet_time = time.time()
        self.lock = threading.Lock()

    def receive_data(self, connection):
        connection.settimeout(self.timeout)
        while True:
            try:
                data = connection.recv(self.buffer_size)
                if not data:
                    break

                with self.lock:
                    self.data_buffer += data
                    self.last_packet_time = time.time()

            except socket.timeout:
                with self.lock:
                    if time.time() - self.last_packet_time > 5 * self.timeout:
                        break
            except Exception as e:
                print(f"Error receiving data: {e}")
                break

    def get_next_message(self):
        with self.lock:
            if b"\n" in self.data_buffer:
                message, _, self.data_buffer = self.data_buffer.partition(b"\n")
                return message.decode()
            else:
                return None

def process_sensor_data(decoded_data, connection):
    # Example sensor data processing (uncomment and modify as needed):
    # if "Left:" in decoded_data:
    #     parts = decoded_data.split(",")
    #     for part in parts:
    #         if "Left:" in part:
    #             left_distance = part.split(":")[1].strip().replace(" cm", "")
    #             print(f"  Left distance: {left_distance}")
    #         elif "Back:" in part:
    #             back_distance = part.split(":")[1].strip().replace(" cm", "")
    #             print(f"  Back distance: {back_distance}")
    #         elif "Right:" in part:
    #             right_distance = part.split(":")[1].strip().replace(" cm", "")
    #             print(f"  Right distance: {right_distance}")
    #         elif "Forward:" in part:
    #             forward_distance = part.split(":")[1].strip().replace(" cm", "")
    #             print(f"  Forward distance: {forward_distance}")
    
    # Send a concise response to avoid excessive communication
    try:
        connection.sendall(b"ACK")  # Send an acknowledgement
    except Exception as e:
        print(f"Error sending ACK: {e}")

def handle_client(connection, client_address):
    print(f"Connection from {client_address}")
    receiver = DataReceiver()
    receive_thread = threading.Thread(target=receiver.receive_data, args=(connection,))
    receive_thread.start()

    try:
        while True:
            message = receiver.get_next_message()
            if message:
                print(f"Received: {message}")
                process_sensor_data(message, connection)
            elif not receive_thread.is_alive():
                break  # Exit if the receive thread has finished
            else:
                time.sleep(0.001)  # Small sleep to yield to other threads

    finally:
        receive_thread.join()  # Wait for the receive thread to finish
        connection.close()
        print(f"Connection from {client_address} closed.")

def get_ip_address():
    """Gets the current machine's IP address."""
    try:
        # Try to get the hostname
        hostname = socket.gethostname()
        # Get the IP address associated with the hostname
        ip_address = socket.gethostbyname(hostname)
        return ip_address
    except socket.gaierror:
        # Handle cases where hostname resolution fails
        try:
            # Try connecting to a known server (e.g., Google's DNS)
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip_address = s.getsockname()[0]
            s.close()
            return ip_address
        except Exception:
            return "Unable to determine IP address"

def main():
    # Get and print the IP address
    ip_address = get_ip_address()
    print(f"Server IP address: {ip_address}")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server_address = ('', 8642) # Bind to all available interfaces on port 8080
    server_socket.bind(server_address)

    server_socket.listen(5)
    print(f"Server listening on port {server_address[1]}") # Print only the port

    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")
    secure_socket = context.wrap_socket(server_socket, server_side=True)

    while True:
        print("Waiting for a connection...")
        connection, client_address = secure_socket.accept()
        client_thread = threading.Thread(target=handle_client, args=(connection, client_address))
        client_thread.start()

if __name__ == "__main__":
    main()