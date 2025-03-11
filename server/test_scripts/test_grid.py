import socket
import ssl
import time
import random

# Configuration
SERVER_IP = "10.13.226.252"  # Change to your server IP
SERVER_PORT = 8642
LEFT_OBJECT_DISTANCE = 100  # cm
RIGHT_OBJECT_DISTANCE = 200  # cm
UPDATE_RATE = 0.1  # seconds between updates

def main():
    print(f"Connecting to {SERVER_IP}:{SERVER_PORT}...")
    
    # Create socket and secure with SSL
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # SSL context - accepts self-signed certificates
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    
    secure_socket = context.wrap_socket(client_socket)
    
    try:
        # Connect to server
        secure_socket.connect((SERVER_IP, SERVER_PORT))
        print("Connected to server!")
        
        # Main loop to send sensor data
        yaw = 0.0
        while True:
            # Simulate some minor variation in sensor readings
            left_variation = random.uniform(-2, 2)
            right_variation = random.uniform(-2, 2)
            
            # Create message in same format as the ESP32 code
            message = (
                f"Left: {LEFT_OBJECT_DISTANCE + left_variation:.1f} cm, "
                f"Back: 501.0 cm, "
                f"Right: {RIGHT_OBJECT_DISTANCE + right_variation:.1f} cm, "
                f"Forward: 501.0 cm, "
                f"Roll: 0.00, Pitch: 0.00, Yaw: {yaw:.2f}"
            )
            
            # Send message with newline
            print(f"Sending: {message}")
            secure_socket.sendall((message + "\n").encode())
            
            # Receive and process response
            try:
                secure_socket.settimeout(0.5)
                response = secure_socket.recv(1024).decode().strip()
                
                if response:
                    print(f"Received: {response}")
                    
                    # Handle motor command
                    if response.startswith("CMD:"):
                        command = response[4:].strip()
                        print(f"Motor command: {command}")
                        
                        # Simulate rotating the robot
                        if command == "left":
                            yaw = (yaw + 5) % 360
                        elif command == "right":
                            yaw = (yaw - 5) % 360
            except socket.timeout:
                pass
            
            # Wait for next update
            time.sleep(UPDATE_RATE)
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        secure_socket.close()
        print("Connection closed")

if __name__ == "__main__":
    main()