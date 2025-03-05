#!/usr/bin/env python3
import socket
import ssl
import threading
import time
import sys
import re
from datetime import datetime

class FastSensorReceiver:
    def __init__(self, port=8642, use_ssl=True):
        self.port = port
        self.use_ssl = use_ssl
        self.message_count = 0
        self.start_time = None
        self.clients = []
        self.running = False
        self.lock = threading.Lock()

    def start_server(self):
        """Start the server socket and listen for connections"""
        # Create server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.server_socket.bind(('0.0.0.0', self.port))
            self.server_socket.listen(5)
            
            # Get server IP for display
            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)
            
            print(f"===== Fast Sensor Receiver =====")
            print(f"Local IP: {ip_address}")
            print(f"Listening on port: {self.port}")
            print(f"SSL enabled: {self.use_ssl}")
            print(f"Waiting for ESP32 to connect...")
            print(f"===================================")
            
            # Wrap with SSL if needed
            if self.use_ssl:
                context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")
                secure_socket = context.wrap_socket(self.server_socket, server_side=True)
                socket_to_use = secure_socket
            else:
                socket_to_use = self.server_socket
            
            self.running = True
            self.start_time = time.time()
            
            # Start stats thread
            stats_thread = threading.Thread(target=self._stats_reporter)
            stats_thread.daemon = True
            stats_thread.start()
            
            # Main accept loop
            while self.running:
                try:
                    client_socket, addr = socket_to_use.accept()
                    print(f"New client connected: {addr}")
                    
                    client_thread = threading.Thread(target=self._handle_client, 
                                                    args=(client_socket, addr))
                    client_thread.daemon = True
                    client_thread.start()
                    
                    with self.lock:
                        self.clients.append((client_socket, addr, client_thread))
                        
                except ssl.SSLError as e:
                    print(f"SSL error during accept: {e}")
                    continue
                except Exception as e:
                    print(f"Error accepting connection: {e}")
                    if not self.running:
                        break
                    continue
                    
        except Exception as e:
            print(f"Server error: {e}")
            self.stop_server()
            
    def stop_server(self):
        """Stop the server and close all connections"""
        self.running = False
        
        # Close all client connections
        with self.lock:
            for client_socket, _, _ in self.clients:
                try:
                    client_socket.close()
                except:
                    pass
            self.clients = []
        
        # Close server socket
        try:
            self.server_socket.close()
        except:
            pass
            
        print("\nServer stopped.")
        
        # Print final stats
        elapsed = time.time() - self.start_time if self.start_time else 0
        if elapsed > 0 and self.message_count > 0:
            rate = self.message_count / elapsed
            print(f"\nFinal statistics:")
            print(f"Total runtime: {elapsed:.2f} seconds")
            print(f"Total messages: {self.message_count}")
            print(f"Average rate: {rate:.2f} messages/second")
    
    def _handle_client(self, client_socket, addr):
        """Handle individual client connection"""
        client_socket.settimeout(1.0)  # 1 second timeout
        buffer = b""
        message_pattern = re.compile(r'Left: (\d+) cm, Back: (\d+) cm, Right: (\d+) cm, Forward: (\d+) cm')
        
        try:
            print(f"Client {addr} handling started")
            while self.running:
                try:
                    # Receive data with timeout
                    data = client_socket.recv(8192)
                    if not data:
                        print(f"Client {addr} disconnected")
                        break
                        
                    # Add to buffer and process lines
                    buffer += data
                    while b"\n" in buffer:
                        message, _, buffer = buffer.partition(b"\n")
                        decoded_message = message.decode().strip()
                        
                        if decoded_message:
                            # Print the message with timestamp
                            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                            
                            # Extract just the values for more compact display
                            match = message_pattern.search(decoded_message)
                            if match:
                                values = match.groups()
                                print(f"[{timestamp}] L:{values[0]} B:{values[1]} R:{values[2]} F:{values[3]} | {decoded_message}")
                            else:
                                print(f"[{timestamp}] {decoded_message}")
                            
                            # Send immediate ACK response (crucial for ESP32)
                            client_socket.sendall(b"ACK\n")
                            
                            # Update message count for stats
                            with self.lock:
                                self.message_count += 1
                                
                except socket.timeout:
                    # This is fine, just a timeout on recv
                    continue
                except Exception as e:
                    print(f"Error handling client {addr}: {e}")
                    break
        finally:
            # Clean up connection
            try:
                client_socket.close()
            except:
                pass
                
            # Remove from clients list
            with self.lock:
                self.clients = [(s, a, t) for s, a, t in self.clients if a != addr]
            
            print(f"Client {addr} handler finished")
    
    def _stats_reporter(self):
        """Periodically report message statistics"""
        last_count = 0
        last_time = time.time()
        
        while self.running:
            time.sleep(5)  # Report every 5 seconds
            
            current_time = time.time()
            with self.lock:
                current_count = self.message_count
            
            # Calculate rate over last interval
            messages_in_period = current_count - last_count
            time_period = current_time - last_time
            
            if time_period > 0:
                rate = messages_in_period / time_period
                total_rate = current_count / (current_time - self.start_time) if self.start_time else 0
                
                print(f"\n--- STATS UPDATE ---")
                print(f"Current rate: {rate:.2f} messages/second")
                print(f"Total messages: {current_count}")
                print(f"Average rate: {total_rate:.2f} messages/second")
                print(f"Connected clients: {len(self.clients)}")
                print(f"-------------------\n")
                
                last_count = current_count
                last_time = current_time

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fast ESP32 Sensor Data Receiver')
    parser.add_argument('--port', type=int, default=8642, help='Port to listen on (default: 8642)')
    parser.add_argument('--disable-ssl', action='store_true', help='Disable SSL/TLS encryption')
    
    args = parser.parse_args()
    
    receiver = FastSensorReceiver(port=args.port, use_ssl=not args.disable_ssl)
    
    try:
        receiver.start_server()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        receiver.stop_server()