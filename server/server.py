import socket
import ssl
import time
import threading
import multiprocessing
import numpy as np
import os
import math
import cv2
import torch
from ultralytics import YOLO
import cvzone
import mediapipe as mp
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import json
import asyncio
import websockets
import sys

from occupancy_grid import OccupancyGrid
from person_tracker import PersonTracker
from ultrasonic_sensor import UltrasonicSensor

# Global constants
CAMERA = 0
FIELD_SIZE_CM = 1250.0
CELL_SIZE_CM = 25.0

# Define sensors (will be initialized in each process that needs them)
def initialize_sensors():
    return {
        'forward': UltrasonicSensor('forward', 0, min_valid_distance=CELL_SIZE_CM),
        'back': UltrasonicSensor('back', 180, min_valid_distance=CELL_SIZE_CM),
        'right': UltrasonicSensor('right', 90, min_valid_distance=CELL_SIZE_CM),
        'left': UltrasonicSensor('left', -90, min_valid_distance=CELL_SIZE_CM)
    }

# ========================
# CAMERA PROCESS
# ========================
def camera_process(shared_frame_data, shared_person_status):
    """Process for camera operations"""
    
    # Initialize camera
    cap = None
    
    # Initialize models and resources
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    
    # Select appropriate device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"[INFO] Camera process using device: {device}")
    
    try:
        # Load YOLO models
        pose_model = YOLO('yolo11n-pose.pt', verbose=False)
        object_model = YOLO('yolo11n.pt', verbose=False)
        
        # Move models to appropriate device
        pose_model.to(device)
        object_model.to(device)
        
        # Initialize MediaPipe face mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        
        person_tracker = PersonTracker()
        
        # Initialize camera
        cap = cv2.VideoCapture(CAMERA)
        if not cap.isOpened():
            print("[ERROR] Could not open camera.")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("[INFO] Camera initialized in camera process")
        
        # Helper functions for processing
        def calculate_EAR(landmarks, eye_indices):
            p2_minus_p6 = np.linalg.norm(landmarks[eye_indices[1]] - landmarks[eye_indices[5]])
            p3_minus_p5 = np.linalg.norm(landmarks[eye_indices[2]] - landmarks[eye_indices[4]])
            p1_minus_p4 = np.linalg.norm(landmarks[eye_indices[0]] - landmarks[eye_indices[3]])
            return (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4 + 1e-6)

        def calculate_head_angles(face_landmarks, image_width, image_height):
            left_eye_outer = face_landmarks.landmark[33]
            right_eye_outer = face_landmarks.landmark[263]
            x1, y1 = left_eye_outer.x * image_width, left_eye_outer.y * image_height
            x2, y2 = right_eye_outer.x * image_width, right_eye_outer.y * image_height
            angle_radians_roll = np.arctan2(y1 - y2, x2 - x1)
            angle_degrees_roll = np.degrees(angle_radians_roll)
            nose_tip = face_landmarks.landmark[1]
            chin = face_landmarks.landmark[152]
            x_nose, y_nose = nose_tip.x * image_width, nose_tip.y * image_height
            x_chin, y_chin = chin.x * image_width, chin.y * image_height
            angle_radians_pitch = np.arctan2(y_chin - y_nose, x_chin - x_nose)
            pitch_angle_adjusted = np.degrees(angle_radians_pitch) - 90
            return angle_degrees_roll, pitch_angle_adjusted

        def determine_pose(keypoints_xy, bbox, head_angles=None, objects_detected=[]):
            # This would normally contain pose determination logic
            return "unknown"
        
        last_frame_time = time.time()
        
        print("[INFO] Camera process initialized and ready")
        
        while True:
            # Calculate time delta for FPS control
            current_time = time.time()
            elapsed = current_time - last_frame_time
            
            # Limit frame processing rate to avoid overloading CPU
            if elapsed < 0.033:  # ~30 FPS max
                time.sleep(0.01)
                continue
                
            last_frame_time = current_time
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to capture frame")
                time.sleep(0.1)
                continue
            
            # Process frame with AI models
            image = frame.copy()
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Pose detection
            detections = []
            pose_results = pose_model(image)
            for result in pose_results:
                boxes = result.boxes
                keypoints_list = result.keypoints
                if boxes is not None and keypoints_list is not None:
                    for idx in range(len(boxes)):
                        box = boxes[idx]
                        keypoint = keypoints_list[idx]
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        keypoints_xy = keypoint.xy.cpu().numpy().squeeze()
                        head_angles = None
                        eye_status = "Eyes Open"
                        face_region = image_rgb[y1:y2, x1:x2]
                        face_results = face_mesh.process(face_region)
                        if face_results.multi_face_landmarks:
                            for face_landmarks in face_results.multi_face_landmarks:
                                h, w = face_region.shape[:2]
                                landmarks = np.array([[int(lm.x * w), int(lm.y * h)] 
                                                    for lm in face_landmarks.landmark])
                                left_ear = calculate_EAR(landmarks, [33, 160, 158, 133, 153, 144])
                                right_ear = calculate_EAR(landmarks, [362, 385, 387, 263, 373, 380])
                                avg_ear = (left_ear + right_ear) / 2.0
                                if avg_ear < 0.25:
                                    eye_status = "Eyes Closed"
                                head_angles = calculate_head_angles(face_landmarks, w, h)
                                break
                        else:
                            eye_status = "Eye Status Unknown"
                        pose_label = determine_pose(keypoints_xy, (x1, y1, x2, y2), head_angles)
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'pose': pose_label,
                            'eye_status': eye_status,
                            'centroid': (cx, cy)
                        })
            
            person_tracker.update(detections)
            
            # Draw trackers
            for track_id, track in person_tracker.tracks.items():
                x1, y1, x2, y2 = track['bbox']
                pose_label = track['data'][-1]['pose']
                eye_status = track['data'][-1]['eye_status']
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cvzone.putTextRect(image, f'ID {track_id}', [x1 + 8, y1 - 40], thickness=2, scale=1)
                cvzone.putTextRect(image, f'Pose: {pose_label}', [x1 + 8, y2 + 30], thickness=2, scale=1)
                cvzone.putTextRect(image, f'Eye Status: {eye_status}', [x1 + 8, y2 + 70], thickness=2, scale=1)
                is_unconscious = person_tracker.check_unconscious(track)
                color = (0, 0, 255) if is_unconscious else (0, 255, 0)
                label = 'Unconscious' if is_unconscious else 'Conscious'
                cvzone.putTextRect(image, label, [x1 + 8, y2 + 110], thickness=2, scale=1, colorR=color)
            
            # Encode the processed frame for streaming
            ret2, buffer = cv2.imencode('.jpg', image)
            if ret2:
                # Put the frame in the shared state
                processed_frame = buffer.tobytes()
                shared_frame_data['current_frame'] = processed_frame
                
                # Extract person status for grid markers
                if person_tracker.tracks:
                    person_status = "Conscious"
                    for track in person_tracker.tracks.values():
                        if person_tracker.check_unconscious(track):
                            person_status = "Unconscious"
                            break
                    # Update shared person status
                    shared_person_status['status'] = person_status
    
    except Exception as e:
        print(f"[CRITICAL] Fatal error in camera process: {e}")
        import traceback
        traceback.print_exc()
        # Try to release camera if there was an error
        if cap is not None:
            try:
                cap.release()
            except:
                pass

# ========================
# GRID PROCESS
# ========================
def grid_process(shared_sensor_data, shared_grid_data, shared_person_status, motor_command):
    """Process to handle occupancy grid updates and person tracking"""
    try:
        # Initialize the occupancy grid
        occupancy_grid = OccupancyGrid(FIELD_SIZE_CM, CELL_SIZE_CM)
        last_update_time = time.time()
        sensors = initialize_sensors()
        last_processed_count = 0
        
        print("[INFO] Grid process initialized")
        
        while True:
            try:
                current_time = time.time()
                
                # Get latest sensor data
                sensor_data = {}
                try:
                    # Get sensor data - only copy what we need, not the whole dictionary
                    if 'latest_msg_count' in shared_sensor_data:
                        current_count = shared_sensor_data['latest_msg_count']
                        
                        # Only process if we have new data
                        if current_count > last_processed_count:
                            last_processed_count = current_count
                            
                            # Get just the fields we need
                            if 'distances' in shared_sensor_data:
                                sensor_data['distances'] = shared_sensor_data['distances'].copy()
                            if 'yaw' in shared_sensor_data:
                                sensor_data['yaw'] = shared_sensor_data['yaw']
                except Exception as e:
                    print(f"[ERROR] Error accessing sensor data: {e}")
                
                # Get current motor command
                current_motor_command = motor_command.get('command', 'stop')
                
                if sensor_data and 'distances' in sensor_data and 'yaw' in sensor_data:
                    # Extract distances and yaw
                    distances = sensor_data.get('distances', {})
                    yaw = sensor_data.get('yaw')
                    
                    if yaw is not None:
                        # Compute relative yaw (no need for calibration offset in simplified version)
                        relative_yaw = yaw
                        
                        # Update sensors with new distances
                        for sensor_name, distance in distances.items():
                            sensors[sensor_name].update_distance(distance)
                        
                        # Prepare sensor data for grid update
                        sensor_data_for_grid = {}
                        for sensor in sensors.values():
                            data = sensor.get_data(relative_yaw)
                            if data:
                                sensor_data_for_grid[sensor.name] = data['distance']
                                sensor_data_for_grid[sensor.name + '_angle'] = data['angle']
                        
                        # Calculate time delta
                        dt = current_time - last_update_time
                        last_update_time = current_time
                        
                        # Update the occupancy grid
                        occupancy_grid.update_from_sensors(sensor_data_for_grid, relative_yaw, dt, current_motor_command)
                
                # Get current grid data
                grid_data = occupancy_grid.get_grid_data()
                
                # Update shared grid data for WebSocket and HTTP processes
                for key, value in grid_data.items():
                    shared_grid_data[key] = value
                
                # Check for person status from camera process
                person_status = shared_person_status.get('status')
                if person_status and occupancy_grid.latest_forward is not None:
                    yaw_rad = math.radians(occupancy_grid.yaw)
                    person_x_cm = occupancy_grid.robot_x + occupancy_grid.latest_forward * math.sin(yaw_rad)
                    person_y_cm = occupancy_grid.robot_y + occupancy_grid.latest_forward * math.cos(yaw_rad)
                    marker_row, marker_col = occupancy_grid.world_to_grid(person_x_cm, person_y_cm)
                    shared_grid_data['person_markers'] = [{"grid_row": marker_row, "grid_col": marker_col, "status": person_status}]
                else:
                    shared_grid_data['person_markers'] = []
                
                # Small sleep to prevent CPU hogging - don't make this longer!
                # Short enough to check for new sensor data frequently
                time.sleep(0.005)  # 5ms sleep - 200Hz check rate
                
            except Exception as e:
                print(f"[ERROR] Grid process error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.5)
    
    except Exception as e:
        print(f"[CRITICAL] Fatal error in grid process: {e}")
        import traceback
        traceback.print_exc()

# ========================
# WEBSOCKET PROCESS
# ========================
def websocket_process(shared_grid_data, motor_command):
    """Process to handle WebSocket communications"""
    try:
        # WebSocket connections set
        websocket_connections = set()
        websocket_lock = threading.Lock()
        
        async def register_websocket(websocket):
            with websocket_lock:
                websocket_connections.add(websocket)
                print(f"[INFO] New WebSocket connection established: {websocket.remote_address}")

        async def websocket_handler(websocket, path):
            await register_websocket(websocket)
            try:
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        if 'command' in data:
                            motor_command['command'] = data['command']
                            print(f"[DEBUG] WebSocket received command: {data['command']}")
                            # Send acknowledgment
                            await websocket.send(json.dumps({"type": "ack", "command": data['command']}))
                    except json.JSONDecodeError:
                        print(f"[ERROR] Invalid JSON received: {message}")
                    except Exception as e:
                        print(f"[ERROR] WebSocket error: {e}")
            finally:
                with websocket_lock:
                    try:
                        websocket_connections.remove(websocket)
                        print(f"[INFO] WebSocket connection closed: {websocket.remote_address}")
                    except KeyError:
                        pass

        async def broadcast_message(message):
            disconnected = set()
            with websocket_lock:
                for websocket in websocket_connections:
                    try:
                        await websocket.send(message)
                    except websockets.exceptions.ConnectionClosed:
                        disconnected.add(websocket)
            
            # Clean up disconnected clients
            if disconnected:
                with websocket_lock:
                    for websocket in disconnected:
                        try:
                            websocket_connections.remove(websocket)
                        except KeyError:
                            pass

        async def broadcast_grid_data():
            """Coroutine to broadcast grid data periodically"""
            while True:
                try:
                    # Create a regular Python dict for JSON serialization
                    data_dict = dict(shared_grid_data)
                    
                    # Create a JSON message
                    message = json.dumps({
                        'type': 'gridData',
                        'data': data_dict
                    })
                    
                    # Broadcast the message
                    await broadcast_message(message)
                    
                    # Use a reasonable update rate for the frontend - 20Hz is good for visualization
                    # This doesn't affect sensor data processing speed
                    await asyncio.sleep(0.05)  # 20Hz update rate
                except Exception as e:
                    print(f"[ERROR] Error broadcasting grid data: {e}")
                    await asyncio.sleep(0.5)

        async def start_websocket_server(host='0.0.0.0', port=8081):
            """Start the WebSocket server"""
            async with websockets.serve(websocket_handler, host, port):
                print(f"WebSocket server started on ws://{host}:{port}")
                # Start the grid data broadcast task
                asyncio.create_task(broadcast_grid_data())
                # Keep the server running
                await asyncio.Future()
        
        # Run the WebSocket server
        asyncio.run(start_websocket_server())
    
    except Exception as e:
        print(f"[CRITICAL] Fatal error in WebSocket process: {e}")
        import traceback
        traceback.print_exc()

# ========================
# HTTP SERVER PROCESS
# ========================
def http_process(shared_frame_data, shared_grid_data):
    """Process to handle HTTP server"""
    try:
        class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed_path = urlparse(self.path)
                path = parsed_path.path

                if parsed_path.path == '/command':
                    query = parse_qs(parsed_path.query)
                    cmd = query.get("cmd", [""])[0]
                    if cmd:
                        motor_command['command'] = cmd
                        self.send_response(200)
                        self.send_header("Content-Type", "text/plain")
                        self.end_headers()
                        self.wfile.write(("Command received: " + cmd).encode())
                    else:
                        self.send_response(400)
                        self.send_header("Content-Type", "text/plain")
                        self.end_headers()
                        self.wfile.write(b"Missing command")
                    return

                if parsed_path.path == '/':
                    self.serve_index_html()
                elif parsed_path.path == '/styles.css':
                    self.serve_css()
                elif parsed_path.path == '/grid':
                    self.serve_grid_json()
                elif parsed_path.path == '/video_feed':
                    self.serve_video_feed()
                else:
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b"Not Found")

            def serve_index_html(self):
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                with open(os.path.join('static', 'index.html'), 'rb') as f:
                    self.wfile.write(f.read())

            def serve_css(self):
                self.send_response(200)
                self.send_header('Content-type', 'text/css')
                self.end_headers()
                with open(os.path.join('static', 'styles.css'), 'rb') as f:
                    self.wfile.write(f.read())

            def serve_grid_json(self):
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                # Create a regular Python dict for JSON serialization
                data_dict = dict(shared_grid_data)
                
                self.wfile.write(json.dumps(data_dict).encode('utf-8'))

            def serve_video_feed(self):
                self.send_response(200)
                self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
                self.end_headers()
                try:
                    while True:
                        # Get latest frame
                        frame_data = shared_frame_data.get('current_frame')
                        
                        if frame_data is None:
                            time.sleep(0.01)
                            continue

                        self.wfile.write(b"--frame\r\n")
                        frame_headers = (
                            "Content-Type: image/jpeg\r\n"
                            "Content-Length: " + str(len(frame_data)) + "\r\n\r\n"
                        )
                        self.wfile.write(frame_headers.encode('utf-8'))
                        self.wfile.write(frame_data)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                        time.sleep(0.033)  # ~30 FPS max
                except Exception as e:
                    print(f"Video feed error: {e}")
        
        # Start HTTP server
        httpd = ThreadingHTTPServer(('0.0.0.0', 8080), SimpleHTTPRequestHandler)
        print(f"HTTP Server listening on http://0.0.0.0:8080")
        
        # This will block until the server is stopped
        httpd.serve_forever()
    
    except Exception as e:
        print(f"[CRITICAL] Fatal error in HTTP process: {e}")
        import traceback
        traceback.print_exc()

# ========================
# OPTIMIZED SENSOR DATA PROCESS
# ========================
def fast_sensor_data_process(shared_sensor_data, motor_command):
    """Optimized process to handle sensor data collection from TCP/SSL server"""
    try:
        # Message counter for tracking updates
        message_count = 0
        message_lock = threading.Lock()
        
        # Statistics tracking
        start_time = time.time()
        last_stats_time = start_time
        last_stats_count = 0
        
        # Sensor data parsing - optimized for speed
        def parse_sensor_data(data):
            parts = data.split(",")
            distances = {}
            yaw = None
            
            for part in parts:
                if ":" not in part:
                    continue
                
                label, value_str = part.split(":", 1)
                label = label.strip().lower()
                
                value_str = value_str.strip()
                if value_str.endswith("cm"):
                    value_str = value_str[:-2].strip()
                
                try:
                    value = float(value_str)
                    
                    if label == 'forward':
                        distances['forward'] = value
                    elif label == 'back':
                        distances['back'] = value
                    elif label == 'right':
                        distances['right'] = value
                    elif label == 'left':
                        distances['left'] = value
                    elif label == 'yaw':
                        yaw = value
                except ValueError:
                    continue
            
            return distances, yaw

        def handle_client(connection, client_address):
            nonlocal message_count, last_stats_time, last_stats_count
            print(f"Connection from {client_address}")
            buffer = b""
            
            # Set socket to non-blocking mode for faster response
            connection.settimeout(0.1)  # Short timeout
            
            try:
                while True:
                    try:
                        data = connection.recv(8192)
                        if not data:
                            break  # Client disconnected
                            
                        buffer += data
                        
                        while b"\n" in buffer:
                            message, _, buffer = buffer.partition(b"\n")
                            decoded_message = message.decode().strip()
                            
                            if decoded_message:
                                distances, yaw = parse_sensor_data(decoded_message)
                                
                                if distances and yaw is not None:
                                    # Update shared sensor data - atomic update
                                    with message_lock:
                                        # Update the shared data dict with our parsed data
                                        shared_sensor_data['distances'] = distances
                                        shared_sensor_data['yaw'] = yaw
                                        message_count += 1
                                        shared_sensor_data['latest_msg_count'] = message_count
                                    
                                    # Get current motor command
                                    current_command = motor_command.get('command', 'stop')
                                    
                                    # Send acknowledgement with motor command
                                    ack_message = f"CMD:{current_command}\n"
                                    connection.sendall(ack_message.encode())
                                    
                                    # Update stats counter
                                    current_time = time.time()
                                    if current_time - last_stats_time >= 5.0:  # Print stats every 5 seconds
                                        process_messages_in_period = message_count - last_stats_count
                                        process_time_elapsed = current_time - last_stats_time
                                        process_rate = process_messages_in_period / process_time_elapsed if process_time_elapsed > 0 else 0
                                        
                                        total_elapsed = current_time - start_time
                                        total_rate = message_count / total_elapsed if total_elapsed > 0 else 0
                                        
                                        print(f"\n--- SENSOR STATS UPDATE ---")
                                        print(f"Current rate: {process_rate:.2f} messages/second")
                                        print(f"Average rate: {total_rate:.2f} messages/second")
                                        print(f"Total messages: {message_count}")
                                        print(f"Last values - Yaw: {yaw:.2f}, Forward: {distances.get('forward', 'N/A')}, "
                                              f"Left: {distances.get('left', 'N/A')}, Right: {distances.get('right', 'N/A')}")
                                        print(f"Last command sent: {current_command}")
                                        print(f"---------------------------\n")
                                        
                                        # Update stats counters
                                        last_stats_time = current_time
                                        last_stats_count = message_count
                                                                                
                    except socket.timeout:
                        # Timeouts are normal, just continue
                        continue
                    except ConnectionResetError:
                        print(f"Connection reset by client {client_address}")
                        break
                    except Exception as e:
                        print(f"Error processing data from {client_address}: {e}")
                        break
            finally:
                connection.close()
                print(f"Connection from {client_address} closed.")

        def get_ip_address():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                ip_address = s.getsockname()[0]
                s.close()
                return ip_address
            except:
                return "127.0.0.1"

        # Start TCP server for sensor data
        ip_address = get_ip_address()
        print(f"--------------------------\nSensor Server IP: {ip_address}\n--------------------------")
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('', 8642))
        server_socket.listen(5)
        print("Sensor SSL server listening on port 8642")
        
        # Initialize SSL
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")
        secure_socket = context.wrap_socket(server_socket, server_side=True)

        # Initialize shared state
        shared_sensor_data['latest_msg_count'] = 0
        shared_sensor_data['distances'] = {}
        shared_sensor_data['yaw'] = 0.0

        while True:
            print("Waiting for a sensor connection...")
            try:
                connection, client_address = secure_socket.accept()
                
                # Set TCP_NODELAY to minimize latency
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                
                # Create and start client handler thread
                client_thread = threading.Thread(target=handle_client, args=(connection, client_address))
                client_thread.daemon = True
                client_thread.start()
            except Exception as e:
                print(f"Error accepting connection: {e}")
                time.sleep(1)  # Brief delay on connection error
    
    except Exception as e:
        print(f"[CRITICAL] Fatal error in fast sensor data process: {e}")
        import traceback
        traceback.print_exc()

# ========================
# MAIN ENTRY POINT
# ========================
def main():
    try:
        # Create manager for shared state across processes
        manager = multiprocessing.Manager()
        
        # Initialize shared state with manager dictionaries
        shared_frame_data = manager.dict()
        shared_grid_data = manager.dict()
        shared_sensor_data = manager.dict()
        shared_person_status = manager.dict()
        motor_command = manager.dict({'command': 'stop'})
        
        print("\n=== Starting Optimized Multiprocessing Server ===\n")
        
        # Start all processes
        processes = []
        
        # Start sensor data process FIRST to ensure it's ready to receive data
        print("Starting Fast Sensor Data Process...")
        sensor_proc = multiprocessing.Process(
            target=fast_sensor_data_process,
            args=(shared_sensor_data, motor_command),
            name="FastSensorDataProcess"
        )
        sensor_proc.daemon = True
        sensor_proc.start()
        processes.append(sensor_proc)
        
        # Give sensor process a moment to initialize
        time.sleep(0.5)
        
        # Start camera process
        print("Starting Camera Process...")
        camera_proc = multiprocessing.Process(
            target=camera_process, 
            args=(shared_frame_data, shared_person_status),
            name="CameraProcess"
        )
        camera_proc.daemon = True
        camera_proc.start()
        processes.append(camera_proc)
        
        # Start grid process
        print("Starting Grid Process...")
        grid_proc = multiprocessing.Process(
            target=grid_process,
            args=(shared_sensor_data, shared_grid_data, shared_person_status, motor_command),
            name="GridProcess"
        )
        grid_proc.daemon = True
        grid_proc.start()
        processes.append(grid_proc)
        
        # Start WebSocket process
        print("Starting WebSocket Process...")
        websocket_proc = multiprocessing.Process(
            target=websocket_process,
            args=(shared_grid_data, motor_command),
            name="WebSocketProcess"
        )
        websocket_proc.daemon = True
        websocket_proc.start()
        processes.append(websocket_proc)
        
        # Start HTTP server process
        print("Starting HTTP Server Process...")
        http_proc = multiprocessing.Process(
            target=http_process,
            args=(shared_frame_data, shared_grid_data),
            name="HTTPProcess"
        )
        http_proc.daemon = True
        http_proc.start()
        processes.append(http_proc)
        
        print("\nAll processes started successfully!")
        print(f"Active processes: {len(processes)}")
        
        # Main loop to keep the process alive
        try:
            while True:
                time.sleep(1)
                
                # Check if any processes have died
                for proc in processes:
                    if not proc.is_alive():
                        print(f"WARNING: Process {proc.name} has died!")
                
        except KeyboardInterrupt:
            print("\nShutting down server...")
        
    except KeyboardInterrupt:
        print("\nShutting down server...")
        # Clean termination of all processes
        for proc in processes:
            proc.terminate()
        print("Server shutdown complete.")
    except Exception as e:
        print(f"Fatal error in main process: {e}")
        # Try to terminate all processes on error
        for proc in processes:
            try:
                proc.terminate()
            except:
                pass
            
if __name__ == "__main__":
    # On macOS, we need this to properly handle multiprocessing
    multiprocessing.freeze_support()
    
    # For macOS, set the start method explicitly
    if sys.platform == 'darwin':
        multiprocessing.set_start_method('spawn')
    
    main()