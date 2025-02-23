import socket
import ssl
import time
import threading
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
import sys

from occupancy_grid import OccupancyGrid
from person_tracker import PersonTracker
from ultrasonic_sensor import UltrasonicSensor

# Global constants
CAMERA = 0
FIELD_SIZE_CM = 1250.0
CELL_SIZE_CM = 10.0

# Global variables for occupancy grid, person markers, and video streaming.
occupancy_grid = OccupancyGrid(FIELD_SIZE_CM, CELL_SIZE_CM)
person_markers = []
latest_frame = None  # JPEG-encoded processed frame for MJPEG streaming

# New globals for decoupled camera capture.
capture_frame = None  # Latest raw frame captured from the camera
capture_lock = threading.Lock()  # Protects access to capture_frame

# yaw calibration offset
calibration_offset = None

# Sensor definitions
sensors = {
    'forward': UltrasonicSensor('forward', 0, min_valid_distance=CELL_SIZE_CM),
    'back': UltrasonicSensor('back', 180, min_valid_distance=CELL_SIZE_CM),
    'right': UltrasonicSensor('right', 90, min_valid_distance=CELL_SIZE_CM),
    'left': UltrasonicSensor('left', -90, min_valid_distance=CELL_SIZE_CM)
}

last_update_time = time.time()
current_client = None  # For sensor connection management
last_motor_command = "stop"

# ------------------------
# Sensor Data Parsing
# ------------------------
def parse_sensor_data(data):
    parts = [p.strip() for p in data.split(",")]
    distances = {}
    yaw = None
    for part in parts:
        if ":" not in part:
            continue
        label, value_str = part.split(":")
        label = label.strip().lower()
        value_str = value_str.strip()
        if value_str.endswith("cm"):
            value_str = value_str[:-2].strip()
        try:
            value = float(value_str)
        except ValueError:
            continue
        if label in sensors:
            distances[label] = value
        elif label == 'yaw':
            yaw = value
    return distances, yaw

# ------------------------
# Data Receiver for Sensor Connection
# ------------------------
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
                    break  # Client disconnected
                with self.lock:
                    self.data_buffer += data
                    self.last_packet_time = time.time()
            except socket.timeout:
                with self.lock:
                    if time.time() - self.last_packet_time > 10 * self.timeout:
                        print("[DEBUG] No data received for an extended period; closing connection.")
                        break
                    else:
                        print("[DEBUG] Socket timeout but connection still active; waiting for data...")
            except Exception as e:
                print(f"Error receiving data: {e}")
                break

    def get_next_message(self):
        with self.lock:
            if b"\n" in self.data_buffer:
                message, _, self.data_buffer = self.data_buffer.partition(b"\n")
                decoded_message = message.decode().strip()
                if not decoded_message:
                    return None
                return decoded_message
            return None

def process_sensor_data(decoded_data, connection):
    global last_update_time, calibration_offset
    distances, yaw = parse_sensor_data(decoded_data)
    print(f"[DEBUG] Raw sensor readings: {distances}, Yaw: {yaw}")
    if yaw is not None:
        # Set the calibration offset on the first valid yaw reading.
        if calibration_offset is None:
            calibration_offset = yaw
            print(f"[DEBUG] Calibration offset set to: {calibration_offset}")
        
        # Compute the relative yaw.
        relative_yaw = (yaw - calibration_offset + 360) % 360

        for sensor_name, distance in distances.items():
            sensors[sensor_name].update_distance(distance)
        sensor_data = {}
        # Use the relative_yaw for sensor angle calculations.
        for sensor in sensors.values():
            data = sensor.get_data(relative_yaw)
            if data:
                sensor_data[sensor.name] = data['distance']
                sensor_data[sensor.name + '_angle'] = data['angle']
                angle_rad = math.radians(data['angle'])
                endpoint_x = occupancy_grid.robot_x + data['distance'] * math.sin(angle_rad)
                endpoint_y = occupancy_grid.robot_y + data['distance'] * math.cos(angle_rad)
                print(f"[DEBUG] Sensor '{sensor.name}': Distance = {data['distance']:.2f} cm, Angle = {data['angle']:.2f}°, Endpoint = ({endpoint_x:.2f}, {endpoint_y:.2f})")
        dt = time.time() - last_update_time
        last_update_time = time.time()
        # Pass the relative yaw to the occupancy grid update.
        occupancy_grid.update_from_sensors(sensor_data, relative_yaw, dt)
        print("[DEBUG] Data Chunk:")
        print(f"  Relative Yaw: {relative_yaw:.2f}°")
        print(f"  Sensor Data: {sensor_data}")
        print(f"  Robot Position: ({occupancy_grid.robot_x:.2f}, {occupancy_grid.robot_y:.2f})")
        connection.sendall(b"ACK")
    else:
        print("Yaw not found in sensor data")


def handle_client(connection, client_address):
    print(f"Connection from {client_address}")
    receiver = DataReceiver()
    receive_thread = threading.Thread(target=receiver.receive_data, args=(connection,))
    receive_thread.start()
    try:
        while True:
            message = receiver.get_next_message()
            if message and message.strip():
                process_sensor_data(message, connection)
            elif not receive_thread.is_alive():
                break
    finally:
        receive_thread.join()
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

# ------------------------
# HTTP Server and Endpoints
# ------------------------
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)

        if parsed_path.path == '/command':
            query = parse_qs(parsed_path.query)
            command = query.get("cmd", [""])[0]
            if command:
                global last_motor_command
                last_motor_command = command
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(("Command received: " + command).encode())
                print(f"[DEBUG] Updated motor command to: {command}")
                return
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
        data = occupancy_grid.get_grid_data()
        data['person_markers'] = person_markers
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def serve_video_feed(self):
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        try:
            while True:
                if latest_frame is None:
                    time.sleep(0.01)
                    continue

                self.wfile.write(b"--frame\r\n")
                frame_headers = (
                    "Content-Type: image/jpeg\r\n"
                    "Content-Length: " + str(len(latest_frame)) + "\r\n\r\n"
                )
                self.wfile.write(frame_headers.encode('utf-8'))
                self.wfile.write(latest_frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except Exception as e:
            print("Video feed error:", e)

def start_http_server(host='127.0.0.1', port=8080):
    httpd = ThreadingHTTPServer((host, port), SimpleHTTPRequestHandler)
    print(f"HTTP Server listening on http://{host}:{port}")
    httpd.serve_forever()

# ------------------------
# Camera Capture and Processing Threads
# ------------------------
# This thread continuously captures raw frames as fast as the camera delivers them.
def camera_capture_thread(cap):
    global capture_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            continue
        with capture_lock:
            capture_frame = frame
        # Capture runs as fast as possible without delay.

# This thread continuously processes the latest captured frame.
def camera_processing_thread():
    global latest_frame, capture_frame, person_markers

    # Initialize models and resources.
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    pose_model = YOLO('yolo11n-pose.pt', verbose=False)
    object_model = YOLO('yolo11n.pt', verbose=False)
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    pose_model.to(device)
    object_model.to(device)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    person_tracker = PersonTracker()

    # Helper functions for processing.
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
        # Insert your pose determination logic here.
        return "unknown"

    while True:
        # Get the most recent captured frame.
        with capture_lock:
            if capture_frame is None:
                continue
            frame = capture_frame.copy()
        image = frame.copy()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # # Object detection.
        # object_results = object_model(image)
        # objects_detected = []
        # for result in object_results:
        #     boxes = result.boxes
        #     if boxes is not None:
        #         for box in boxes:
        #             coords = box.xyxy[0].cpu().numpy().astype(int)
        #             x1_obj, y1_obj, x2_obj, y2_obj = coords
        #             confidence_obj = box.conf[0].cpu().numpy() * 100
        #             class_idx_obj = int(box.cls[0])
        #             class_name_obj = result.names[class_idx_obj]
        #             objects_detected.append({
        #                 'name': class_name_obj,
        #                 'bbox': (x1_obj, y1_obj, x2_obj, y2_obj),
        #                 'confidence': confidence_obj
        #             })
        #             cv2.rectangle(image, (x1_obj, y1_obj), (x2_obj, y2_obj), (255, 0, 0), 2)
        #             cvzone.putTextRect(image, f'{class_name_obj} {confidence_obj:.1f}%', 
        #                                [x1_obj + 5, y1_obj - 10], thickness=1, scale=1, colorR=(255, 0, 0))

        # Pose detection.
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

        # Draw trackers.
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

        # Update person markers from sensor data.
        if person_tracker.tracks and occupancy_grid.latest_forward is not None:
            status = "Conscious"
            for track in person_tracker.tracks.values():
                if person_tracker.check_unconscious(track):
                    status = "Unconscious"
                    break
            yaw_rad = math.radians(occupancy_grid.yaw)
            person_x_cm = occupancy_grid.robot_x + occupancy_grid.latest_forward * math.sin(yaw_rad)
            person_y_cm = occupancy_grid.robot_y + occupancy_grid.latest_forward * math.cos(yaw_rad)
            marker_row, marker_col = occupancy_grid.world_to_grid(person_x_cm, person_y_cm)
            person_markers[:] = [{"grid_row": marker_row, "grid_col": marker_col, "status": status}]
        else:
            person_markers[:] = []

        # Encode the processed frame for streaming.
        ret2, buffer = cv2.imencode('.jpg', image)
        if ret2:
            latest_frame = buffer.tobytes()
        # Loop continuously with no artificial delay.

def camera_loop():
    cap = cv2.VideoCapture(CAMERA)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Start the capture and processing threads.
    capture_thread = threading.Thread(target=camera_capture_thread, args=(cap,), daemon=True)
    processing_thread = threading.Thread(target=camera_processing_thread, daemon=True)
    capture_thread.start()
    processing_thread.start()
    capture_thread.join()
    processing_thread.join()

# ------------------------
# Main Entry Point
# ------------------------
def main():
    global current_client
    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()
    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    camera_thread.start()

    ip_address = get_ip_address()
    print(f"--------------------------\nSensor Server IP: {ip_address}\n--------------------------")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('', 8642))
    server_socket.listen(5)
    print("Sensor SSL server listening on port 8642")
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")
    secure_socket = context.wrap_socket(server_socket, server_side=True)

    while True:
        print("Waiting for a sensor connection...")
        connection, client_address = secure_socket.accept()
        if current_client is not None:
            print("[DEBUG] New client connection detected. Closing previous connection.")
            try:
                current_client.shutdown(socket.SHUT_RDWR)
                current_client.close()
            except Exception as e:
                print(f"[DEBUG] Error closing old client: {e}")
        current_client = connection
        client_thread = threading.Thread(target=handle_client, args=(connection, client_address))
        client_thread.start()

# --- Test functions ---
def run_tests():
    print("Running tests for sensor data processing...")
    
    class FakeConnection:
        def sendall(self, data):
            print("[FakeConnection] Sent data:", data)
    
    fake_conn = FakeConnection()
    
    test_data1 = "forward: 100cm, back: 150cm, right: 75cm, left: 80cm, yaw: 45"
    print("\n[Test 1] Input:", test_data1)
    process_sensor_data(test_data1, fake_conn)
    
    test_data2 = "forward: 100cm, back: 150cm, right: 75cm, left: 80cm"
    print("\n[Test 2] Input:", test_data2)
    process_sensor_data(test_data2, fake_conn)
    
    test_data3 = "forward: 600cm, back: 10cm, right: 75cm, left: 80cm, yaw: 90"
    print("\n[Test 3] Input:", test_data3)
    process_sensor_data(test_data3, fake_conn)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
    else:
        main()
