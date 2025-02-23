#!/usr/bin/env python3
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
from urllib.parse import urlparse

# ================= Global Variables =================
CAMERA = 0
FIELD_SIZE_CM = 1250.0  # Field dimensions in cm (1250 x 1250)
CELL_SIZE_CM = 25.0     # Each grid cell is 25 cm
GRID_ROWS = int(FIELD_SIZE_CM / CELL_SIZE_CM)
GRID_COLS = int(FIELD_SIZE_CM / CELL_SIZE_CM)
log_odds_grid = np.full((GRID_ROWS, GRID_COLS), -2.0, dtype=float)

# amount of time until it says theyre unconscious
DETERMINATION_TIME = 1

# Robot’s current (global) position in centimeters.
robot_x = 0.0
robot_y = 0.0

# Variables used to limit “speed”
last_x = 0.0
last_y = 0.0
last_update_time = time.time()
MAX_SPEED_CM_S = 50.0

# Tuning parameters for updating occupancy
LOG_ODDS_OCCUPIED_INC = 0.9
LOG_ODDS_FREE_DEC = 0.4
MAX_LOG_ODDS = 5.0
MIN_LOG_ODDS = -5.0

# Globals for the camera/vision system
latest_frame = None           # Latest JPEG–encoded video frame
person_markers = []           # List of markers to overlay on the grid (see below)
latest_forward = None         # Latest forward sensor reading (in cm)

# ================= Helper Functions for Occupancy Grid =================
def clamp_log_odds(value):
    return max(MIN_LOG_ODDS, min(MAX_LOG_ODDS, value))

def log_odds_to_probability(l):
    return 1.0 / (1.0 + np.exp(-l))

def update_cell_log_odds(row, col, delta):
    if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
        current = log_odds_grid[row, col]
        new_val = clamp_log_odds(current + delta)
        log_odds_grid[row, col] = new_val

def mark_line_as_free_and_end_as_occupied(row0, col0, d_row, d_col, distance_cells):
    steps = int(distance_cells)
    if steps < 1:
        end_r = row0 + d_row
        end_c = col0 + d_col
        update_cell_log_odds(end_r, end_c, LOG_ODDS_OCCUPIED_INC)
        return
    for i in range(1, steps):
        r = row0 + i * d_row
        c = col0 + i * d_col
        update_cell_log_odds(r, c, -LOG_ODDS_FREE_DEC)
    end_r = row0 + steps * d_row
    end_c = col0 + steps * d_col
    update_cell_log_odds(end_r, end_c, LOG_ODDS_OCCUPIED_INC)

def ignore_or_use_distance(d):
    if d is None:
        return None
    if d > 500 or d <= CELL_SIZE_CM:
        return None
    return d

def world_to_grid(x_cm, y_cm):
    row = int(y_cm // CELL_SIZE_CM)
    col = int(x_cm // CELL_SIZE_CM)
    row = max(0, min(GRID_ROWS - 1, row))
    col = max(0, min(GRID_COLS - 1, col))
    return row, col

def get_robot_cell():
    return world_to_grid(robot_x, robot_y)

def update_robot_position(f, b, r, l):
    global robot_x, robot_y, last_x, last_y, last_update_time
    now = time.time()
    dt = now - last_update_time
    if dt <= 0:
        dt = 0.000001

    old_x = robot_x
    old_y = robot_y
    new_x = old_x
    new_y = old_y

    validF = (f is not None)
    validB = (b is not None)
    if validF and validB:
        new_y = ((FIELD_SIZE_CM - f) + b) / 2.0
    elif validF:
        new_y = FIELD_SIZE_CM - f
    elif validB:
        new_y = b

    validR = (r is not None)
    validL = (l is not None)
    if validR and validL:
        new_x = ((FIELD_SIZE_CM - r) + l) / 2.0
    elif validR:
        new_x = FIELD_SIZE_CM - r
    elif validL:
        new_x = l

    if new_x < 0: new_x = 0
    if new_x > FIELD_SIZE_CM: new_x = FIELD_SIZE_CM
    if new_y < 0: new_y = 0
    if new_y > FIELD_SIZE_CM: new_y = FIELD_SIZE_CM

    dx = new_x - old_x
    dy = new_y - old_y
    dist = math.sqrt(dx*dx + dy*dy)
    max_travel = MAX_SPEED_CM_S * dt

    if dist <= max_travel:
        robot_x = new_x
        robot_y = new_y
        last_x = new_x
        last_y = new_y
        last_update_time = now

def update_occupancy_grid(sensor_data):
    global robot_x, robot_y, latest_forward
    parts = [p.strip() for p in sensor_data.split(",")]
    local_forward = None
    local_back = None
    local_right = None
    local_left = None

    for part in parts:
        if ":" not in part:
            continue
        label, dist_str = part.split(":")
        label = label.strip().lower()
        dist_str = dist_str.strip()
        if dist_str.endswith("cm"):
            dist_str = dist_str[:-2].strip()
        try:
            val = float(dist_str)
        except ValueError:
            continue
        if label == "forward":
            local_forward = val
            latest_forward = val  # update global forward reading
        elif label == "back":
            local_back = val
        elif label == "right":
            local_right = val
        elif label == "left":
            local_left = val

    f = ignore_or_use_distance(local_forward)
    b = ignore_or_use_distance(local_back)
    r = ignore_or_use_distance(local_right)
    l = ignore_or_use_distance(local_left)

    old_x = robot_x
    old_y = robot_y
    update_robot_position(f, b, r, l)
    if (robot_x == old_x) and (robot_y == old_y):
        return

    vantage_row, vantage_col = get_robot_cell()
    if f is not None:
        dist_cells = f / CELL_SIZE_CM
        mark_line_as_free_and_end_as_occupied(vantage_row, vantage_col, +1, 0, dist_cells)
    if b is not None:
        dist_cells = b / CELL_SIZE_CM
        mark_line_as_free_and_end_as_occupied(vantage_row, vantage_col, -1, 0, dist_cells)
    if r is not None:
        dist_cells = r / CELL_SIZE_CM
        mark_line_as_free_and_end_as_occupied(vantage_row, vantage_col, 0, +1, dist_cells)
    if l is not None:
        dist_cells = l / CELL_SIZE_CM
        mark_line_as_free_and_end_as_occupied(vantage_row, vantage_col, 0, -1, dist_cells)

# ================= Sensor Server Code (unchanged) =================

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
    print("Processing sensor data:", decoded_data)
    update_occupancy_grid(decoded_data)
    try:
        connection.sendall(b"ACK")
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
                break
    finally:
        receive_thread.join()
        connection.close()
        print(f"Connection from {client_address} closed.")

def get_ip_address():
    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        return ip_address
    except:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip_address = s.getsockname()[0]
            s.close()
            return ip_address
        except:
            return "Unable to determine IP address"

# ================= HTTP Server Code (with new /video_feed endpoint) =================

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        if parsed_path.path == '/':
            self.serve_index_html()
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
        file_path = os.path.join('static', 'index.html')
        with open(file_path, 'rb') as f:
            self.wfile.write(f.read())

    def serve_grid_json(self):
        import json
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        prob_grid = log_odds_to_probability(log_odds_grid)
        prob_list = prob_grid.tolist()
        r, c = get_robot_cell()
        data = {
            "grid": prob_list,
            "robot_x": robot_x,
            "robot_y": robot_y,
            "robot_row": r,
            "robot_col": c,
            "person_markers": person_markers  # Added person marker info here
        }
        json_data = json.dumps(data)
        self.wfile.write(json_data.encode('utf-8'))

    def serve_video_feed(self):
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        try:
            while True:
                global latest_frame
                if latest_frame is None:
                    time.sleep(0.1)
                    continue
                self.wfile.write(b"--frame\r\n")
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(latest_frame)))
                self.end_headers()
                self.wfile.write(latest_frame)
                self.wfile.write(b"\r\n")
                time.sleep(0.1)
        except Exception as e:
            # When the client disconnects, an exception will be thrown
            pass

def start_http_server(host='127.0.0.1', port=8080):
    httpd = ThreadingHTTPServer((host, port), SimpleHTTPRequestHandler)
    print(f"HTTP Server listening on http://{host}:{port}")
    httpd.serve_forever()

# ================= Person Detection / Vision Code =================
# (Modified from determine_conscious.py – note that cv2.imshow and waitKey have been removed)

# ---- Helper Functions from determine_conscious.py ----

POSE_CONFIDENCE_THRESHOLD = 0.1  # 10% difference
HEAD_PITCH_THRESHOLD = 15  # Degrees
HEAD_ROLL_THRESHOLD = 20  # Degrees
HEAD_SIDEWAYS_THRESHOLD = 45  # Degrees
ASPECT_RATIO_THRESHOLD = 1.5
MIN_BBOX_HEIGHT = 100
WEIGHT_HEAD_POSITION = 2
WEIGHT_HEAD_ORIENTATION = 3
WEIGHT_BODY_ALIGNMENT = 2
WEIGHT_ARM_POSITION = 1
WEIGHT_LEG_ANGLES = 1
WEIGHT_BBOX_ASPECT = 1
WEIGHT_OBJECT_NEARBY = 2
ARM_DISTANCE_THRESHOLD = 50  # Pixels
KNEE_ANGLE_BENT = 120  # Degrees
Y_RANGE_THRESHOLD = 150  # Pixels
VOTE_DIFF_THRESHOLD = 0

EAR_THRESHOLD = 0.25  # Threshold for eye closure

def calculate_EAR(landmarks, eye_indices):
    p2_minus_p6 = np.linalg.norm(landmarks[eye_indices[1]] - landmarks[eye_indices[5]])
    p3_minus_p5 = np.linalg.norm(landmarks[eye_indices[2]] - landmarks[eye_indices[4]])
    p1_minus_p4 = np.linalg.norm(landmarks[eye_indices[0]] - landmarks[eye_indices[3]])
    ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4 + 1e-6)
    return ear

LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

def calculate_head_angles(face_landmarks, image_width, image_height):
    left_eye_outer = face_landmarks.landmark[33]
    right_eye_outer = face_landmarks.landmark[263]
    x1 = left_eye_outer.x * image_width
    y1 = left_eye_outer.y * image_height
    x2 = right_eye_outer.x * image_width
    y2 = right_eye_outer.y * image_height
    angle_radians_roll = np.arctan2(y1 - y2, x2 - x1)
    angle_degrees_roll = np.degrees(angle_radians_roll)
    nose_tip = face_landmarks.landmark[1]
    chin = face_landmarks.landmark[152]
    x_nose = nose_tip.x * image_width
    y_nose = nose_tip.y * image_height
    x_chin = chin.x * image_width
    y_chin = chin.y * image_height
    angle_radians_pitch = np.arctan2(y_chin - y_nose, x_chin - x_nose)
    angle_degrees_pitch = np.degrees(angle_radians_pitch)
    pitch_angle_adjusted = angle_degrees_pitch - 90
    return angle_degrees_roll, pitch_angle_adjusted

def is_valid(kp):
    return not np.array_equal(kp, [0, 0])

def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle

def determine_pose(keypoints_xy, bbox, head_angles=None, objects_detected=[]):
    # Keypoint indices (for brevity)
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    votes = {'Standing': 0, 'Sitting': 0, 'Lying Down': 0}

    if is_valid(keypoints_xy[NOSE]) and (is_valid(keypoints_xy[LEFT_SHOULDER]) or is_valid(keypoints_xy[RIGHT_SHOULDER])):
        nose_y = keypoints_xy[NOSE][1]
        shoulders_y = []
        if is_valid(keypoints_xy[LEFT_SHOULDER]):
            shoulders_y.append(keypoints_xy[LEFT_SHOULDER][1])
        if is_valid(keypoints_xy[RIGHT_SHOULDER]):
            shoulders_y.append(keypoints_xy[RIGHT_SHOULDER][1])
        if shoulders_y:
            avg_shoulder_y = np.mean(shoulders_y)
            if nose_y < avg_shoulder_y:
                votes['Standing'] += WEIGHT_HEAD_POSITION
                votes['Sitting'] += WEIGHT_HEAD_POSITION
            else:
                votes['Lying Down'] += WEIGHT_HEAD_POSITION

    if head_angles is not None:
        head_roll_angle, head_pitch_angle = head_angles
        if abs(head_roll_angle) < HEAD_ROLL_THRESHOLD and abs(head_pitch_angle) < HEAD_PITCH_THRESHOLD:
            votes['Standing'] += WEIGHT_HEAD_ORIENTATION
            votes['Sitting'] += WEIGHT_HEAD_ORIENTATION
        elif abs(head_roll_angle) > HEAD_SIDEWAYS_THRESHOLD or abs(head_pitch_angle) > HEAD_SIDEWAYS_THRESHOLD:
            votes['Lying Down'] += WEIGHT_HEAD_ORIENTATION
        else:
            votes['Sitting'] += WEIGHT_HEAD_ORIENTATION

    keypoints_y = []
    keypoints_indices = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]
    for idx in keypoints_indices:
        if is_valid(keypoints_xy[idx]):
            keypoints_y.append(keypoints_xy[idx][1])
    num_upper_body_kps = sum(is_valid(keypoints_xy[idx]) for idx in [LEFT_SHOULDER, RIGHT_SHOULDER])
    num_lower_body_kps = sum(is_valid(keypoints_xy[idx]) for idx in [LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE])
    if keypoints_y and num_upper_body_kps > 0 and num_lower_body_kps > 0:
        y_range = np.max(keypoints_y) - np.min(keypoints_y)
        if y_range > Y_RANGE_THRESHOLD:
            votes['Standing'] += WEIGHT_BODY_ALIGNMENT
            votes['Sitting'] += WEIGHT_BODY_ALIGNMENT
        else:
            votes['Lying Down'] += WEIGHT_BODY_ALIGNMENT

    elbow_angles = []
    if is_valid(keypoints_xy[LEFT_SHOULDER]) and is_valid(keypoints_xy[LEFT_ELBOW]) and is_valid(keypoints_xy[LEFT_WRIST]):
        angle = calculate_angle(keypoints_xy[LEFT_SHOULDER], keypoints_xy[LEFT_ELBOW], keypoints_xy[LEFT_WRIST])
        elbow_angles.append(angle)
    if is_valid(keypoints_xy[RIGHT_SHOULDER]) and is_valid(keypoints_xy[RIGHT_ELBOW]) and is_valid(keypoints_xy[RIGHT_WRIST]):
        angle = calculate_angle(keypoints_xy[RIGHT_SHOULDER], keypoints_xy[RIGHT_ELBOW], keypoints_xy[RIGHT_WRIST])
        elbow_angles.append(angle)
    if elbow_angles:
        avg_elbow_angle = np.mean(elbow_angles)
        if avg_elbow_angle < 160:
            votes['Sitting'] += WEIGHT_ARM_POSITION
        else:
            votes['Standing'] += WEIGHT_ARM_POSITION
            votes['Lying Down'] += WEIGHT_ARM_POSITION

    knee_angles = []
    if is_valid(keypoints_xy[LEFT_HIP]) and is_valid(keypoints_xy[LEFT_KNEE]) and is_valid(keypoints_xy[LEFT_ANKLE]):
        angle = calculate_angle(keypoints_xy[LEFT_HIP], keypoints_xy[LEFT_KNEE], keypoints_xy[LEFT_ANKLE])
        knee_angles.append(angle)
    if is_valid(keypoints_xy[RIGHT_HIP]) and is_valid(keypoints_xy[RIGHT_KNEE]) and is_valid(keypoints_xy[RIGHT_ANKLE]):
        angle = calculate_angle(keypoints_xy[RIGHT_HIP], keypoints_xy[RIGHT_KNEE], keypoints_xy[RIGHT_ANKLE])
        knee_angles.append(angle)
    if knee_angles:
        avg_knee_angle = np.mean(knee_angles)
        if avg_knee_angle < KNEE_ANGLE_BENT:
            votes['Sitting'] += WEIGHT_LEG_ANGLES * 2
        else:
            votes['Standing'] += WEIGHT_LEG_ANGLES
            votes['Lying Down'] += WEIGHT_LEG_ANGLES

    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    if bbox_height >= MIN_BBOX_HEIGHT:
        aspect_ratio = bbox_width / bbox_height
        if aspect_ratio > ASPECT_RATIO_THRESHOLD:
            votes['Lying Down'] += WEIGHT_BBOX_ASPECT
        else:
            votes['Standing'] += WEIGHT_BBOX_ASPECT
            votes['Sitting'] += WEIGHT_BBOX_ASPECT

    valid_keypoints = [idx for idx in range(len(keypoints_xy)) if is_valid(keypoints_xy[idx])]
    if len(valid_keypoints) <= 5:
        if is_valid(keypoints_xy[NOSE]):
            if head_angles is not None and abs(head_angles[0]) < HEAD_ROLL_THRESHOLD:
                votes['Standing'] += WEIGHT_HEAD_POSITION
                votes['Sitting'] += WEIGHT_HEAD_POSITION
            elif head_angles is not None and abs(head_angles[0]) > HEAD_SIDEWAYS_THRESHOLD:
                votes['Lying Down'] += WEIGHT_HEAD_POSITION
            else:
                votes['Sitting'] += WEIGHT_HEAD_POSITION

    person_bottom_y = max(
        y2,
        keypoints_xy[LEFT_HIP][1] if is_valid(keypoints_xy[LEFT_HIP]) else 0,
        keypoints_xy[RIGHT_HIP][1] if is_valid(keypoints_xy[RIGHT_HIP]) else 0,
        keypoints_xy[LEFT_KNEE][1] if is_valid(keypoints_xy[LEFT_KNEE]) else 0,
        keypoints_xy[RIGHT_KNEE][1] if is_valid(keypoints_xy[RIGHT_KNEE]) else 0
    )
    for obj in objects_detected:
        if obj['name'] in ['chair', 'sofa', 'bench']:
            chair_bbox = obj['bbox']
            chair_top_y = chair_bbox[1]
            if abs(chair_top_y - person_bottom_y) < 100:
                votes['Sitting'] += WEIGHT_OBJECT_NEARBY * 2

    if (is_valid(keypoints_xy[LEFT_ANKLE]) or is_valid(keypoints_xy[RIGHT_ANKLE])) and \
       (is_valid(keypoints_xy[LEFT_HIP]) or is_valid(keypoints_xy[RIGHT_HIP])):
        ankles_y = []
        hips_y = []
        if is_valid(keypoints_xy[LEFT_ANKLE]):
            ankles_y.append(keypoints_xy[LEFT_ANKLE][1])
        if is_valid(keypoints_xy[RIGHT_ANKLE]):
            ankles_y.append(keypoints_xy[RIGHT_ANKLE][1])
        if is_valid(keypoints_xy[LEFT_HIP]):
            hips_y.append(keypoints_xy[LEFT_HIP][1])
        if is_valid(keypoints_xy[RIGHT_HIP]):
            hips_y.append(keypoints_xy[RIGHT_HIP][1])
        if ankles_y and hips_y:
            avg_ankle_y = np.mean(ankles_y)
            avg_hip_y = np.mean(hips_y)
            if abs(avg_ankle_y - avg_hip_y) < 50:
                votes['Sitting'] += WEIGHT_BODY_ALIGNMENT * 2
                votes['Lying Down'] += WEIGHT_BODY_ALIGNMENT * 2

    total_votes = sum(votes.values())
    if total_votes == 0:
        return 'Pose Unknown'
    sorted_votes = sorted(votes.items(), key=lambda item: item[1], reverse=True)
    max_votes = sorted_votes[0][1]
    second_max_votes = sorted_votes[1][1]
    percentage_diff = (max_votes - second_max_votes) / total_votes
    if percentage_diff <= POSE_CONFIDENCE_THRESHOLD:
        possible_poses = [pose for pose, vote in votes.items() if vote == max_votes or vote == second_max_votes]
        final_pose = ' or '.join(possible_poses)
    else:
        final_pose = sorted_votes[0][0]
    return final_pose

# ---- Person Tracker Class ----
class PersonTracker:
    def __init__(self):
        self.tracks = {}  # track_id: track_info
        self.next_id = 1
        self.max_age = 5  # frames
        self.max_distance = 50  # pixels

    def update(self, detections):
        updated_tracks = {}
        for detection in detections:
            detection_centroid = detection['centroid']
            min_distance = float('inf')
            matched_track_id = None
            for track_id, track in self.tracks.items():
                track_centroid = track['centroid']
                distance = np.linalg.norm(np.array(detection_centroid) - np.array(track_centroid))
                if distance < self.max_distance and distance < min_distance:
                    min_distance = distance
                    matched_track_id = track_id
            if matched_track_id is not None:
                track = self.tracks[matched_track_id]
                track['bbox'] = detection['bbox']
                track['centroid'] = detection['centroid']
                track['age'] = 0
                timestamp = time.time()
                track['data'].append({
                    'timestamp': timestamp,
                    'pose': detection['pose'],
                    'eye_status': detection['eye_status']
                })
                self.trim_data(track)
                updated_tracks[matched_track_id] = track
            else:
                track_id = self.next_id
                self.next_id += 1
                timestamp = time.time()
                track = {
                    'id': track_id,
                    'bbox': detection['bbox'],
                    'centroid': detection['centroid'],
                    'age': 0,
                    'start_time': timestamp,
                    'data': [{
                        'timestamp': timestamp,
                        'pose': detection['pose'],
                        'eye_status': detection['eye_status']
                    }],
                    'unconscious': False,
                    'unconscious_since': None
                }
                updated_tracks[track_id] = track
        for track_id in self.tracks.keys():
            if track_id not in updated_tracks:
                track = self.tracks[track_id]
                track['age'] += 1
                if track['age'] <= self.max_age:
                    updated_tracks[track_id] = track
        self.tracks = {tid: t for tid, t in updated_tracks.items() if t['age'] <= self.max_age}

    def trim_data(self, track):
        current_time = time.time()
        track['data'] = [entry for entry in track['data'] if current_time - entry['timestamp'] <= DETERMINATION_TIME]

    def check_unconscious(self, track):
        data = track['data']
        num_entries = len(data)
        if num_entries == 0:
            return False
        poses = [entry['pose'] for entry in data]
        eye_statuses = [entry['eye_status'] for entry in data]
        sitting_or_lying = [pose in ['Sitting', 'Lying Down'] for pose in poses]
        eyes_closed = [eye_status == 'Eyes Closed' for eye_status in eye_statuses]
        percent_sitting_or_lying = sum(sitting_or_lying) / num_entries
        percent_eyes_closed = sum(eyes_closed) / num_entries
        unconscious_criteria_met = percent_sitting_or_lying >= 0.8 and percent_eyes_closed >= 0.8
        current_time = time.time()
        if unconscious_criteria_met:
            if not track.get('unconscious'):
                track['unconscious'] = True
                track['unconscious_since'] = current_time
        else:
            if track.get('unconscious'):
                time_unconscious = current_time - track['unconscious_since']
                if time_unconscious >= 10:
                    percent_conscious_pose = 1 - percent_sitting_or_lying
                    percent_eyes_open = 1 - percent_eyes_closed
                    conscious_criteria_met = percent_conscious_pose >= 0.8 and percent_eyes_open >= 0.8
                    if conscious_criteria_met:
                        track['unconscious'] = False
                        track['unconscious_since'] = None
                else:
                    pass
            else:
                track['unconscious'] = False
                track['unconscious_since'] = None
        return track['unconscious']

# ---- Main Camera Loop ----
def camera_loop():
    global latest_frame, person_markers, robot_x, robot_y, latest_forward
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    # Load models
    pose_model = YOLO('yolo11n-pose.pt', verbose=False)
    object_model = YOLO('yolo11n.pt', verbose=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pose_model.to(device)
    object_model.to(device)
    # Initialize camera
    cap = cv2.VideoCapture(CAMERA)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    person_tracker = PersonTracker()
    while True:
        ret, image = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            # exit()
            break
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # ---- Object Detection ----
        object_results = object_model(image)
        objects_detected = []
        for result in object_results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    x1_obj, y1_obj, x2_obj, y2_obj = coords
                    confidence_obj = box.conf[0].cpu().numpy() * 100
                    class_idx_obj = int(box.cls[0])
                    class_name_obj = result.names[class_idx_obj]
                    objects_detected.append({
                        'name': class_name_obj,
                        'bbox': (x1_obj, y1_obj, x2_obj, y2_obj),
                        'confidence': confidence_obj
                    })
                    cv2.rectangle(image, (x1_obj, y1_obj), (x2_obj, y2_obj), (255, 0, 0), 2)
                    cvzone.putTextRect(image, f'{class_name_obj} {confidence_obj:.1f}%', [x1_obj + 5, y1_obj - 10],
                                       thickness=1, scale=1, colorR=(255, 0, 0))
        # ---- Pose Detection ----
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
                    face_region = image_rgb[y1:y2, x1:x2]
                    face_results = face_mesh.process(face_region)
                    eye_status = "Eyes Open"
                    if face_results.multi_face_landmarks:
                        for face_landmarks in face_results.multi_face_landmarks:
                            h, w, _ = face_region.shape
                            landmarks = []
                            for lm in face_landmarks.landmark:
                                landmarks.append([int(lm.x * w), int(lm.y * h)])
                            landmarks = np.array(landmarks)
                            left_ear = calculate_EAR(landmarks, LEFT_EYE_INDICES)
                            right_ear = calculate_EAR(landmarks, RIGHT_EYE_INDICES)
                            avg_ear = (left_ear + right_ear) / 2.0
                            if avg_ear < EAR_THRESHOLD:
                                eye_status = "Eyes Closed"
                            head_roll_angle, head_pitch_angle = calculate_head_angles(face_landmarks, w, h)
                            head_angles = (head_roll_angle, head_pitch_angle)
                            break
                    else:
                        eye_status = "Eye Status Unknown"
                    pose_label = determine_pose(keypoints_xy, (x1, y1, x2, y2), head_angles, objects_detected)
                    detection = {
                        'bbox': (x1, y1, x2, y2),
                        'pose': pose_label,
                        'eye_status': eye_status,
                        'centroid': (cx, cy)
                    }
                    detections.append(detection)
        person_tracker.update(detections)
        # ---- Draw Trackers ----
        for track_id, track in person_tracker.tracks.items():
            x1, y1, x2, y2 = track['bbox']
            pose_label = track['data'][-1]['pose']
            eye_status = track['data'][-1]['eye_status']
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cvzone.putTextRect(image, f'ID {track_id}', [x1 + 8, y1 - 40], thickness=2, scale=1)
            cvzone.putTextRect(image, f'Pose: {pose_label}', [x1 + 8, y2 + 30], thickness=2, scale=1)
            cvzone.putTextRect(image, f'Eye Status: {eye_status}', [x1 + 8, y2 + 70], thickness=2, scale=1)
            is_unconscious = person_tracker.check_unconscious(track)
            if is_unconscious:
                cvzone.putTextRect(image, 'Unconscious', [x1 + 8, y2 + 110], thickness=2, scale=1, colorR=(0, 0, 255))
            else:
                cvzone.putTextRect(image, 'Conscious', [x1 + 8, y2 + 110], thickness=2, scale=1, colorR=(0, 255, 0))
        # ---- Update Global Person Marker ----
        # For simplicity, if any track is present and we have a valid forward sensor reading,
        # assume the person is directly in front of the robot by that distance.
        if person_tracker.tracks and latest_forward is not None:
            status = "Conscious"
            for track in person_tracker.tracks.values():
                if person_tracker.check_unconscious(track):
                    status = "Unconscious"
                    break
            # Assume forward direction is along +row: person position = robot position + latest_forward in cm.
            person_x_cm = robot_x
            person_y_cm = robot_y + latest_forward
            marker_row, marker_col = world_to_grid(person_x_cm, person_y_cm)
            person_markers[:] = [{"grid_row": marker_row, "grid_col": marker_col, "status": status}]
        else:
            person_markers[:] = []
        # ---- Encode Frame for Video Feed ----
        ret2, buffer = cv2.imencode('.jpg', image)
        if ret2:
            latest_frame = buffer.tobytes()
        time.sleep(0.05)

# ================= Main Entry Point =================
def main():
    # Start the HTTP server in a background thread.
    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()
    # Start the camera/vision thread.
    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    camera_thread.start()

    ip_address = get_ip_address()
    print(f"Sensor Server IP: {ip_address}")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_address = ('', 8642)
    server_socket.bind(server_address)
    server_socket.listen(5)
    print(f"Sensor SSL server listening on {server_address}")

    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")
    secure_socket = context.wrap_socket(server_socket, server_side=True)

    while True:
        print("Waiting for a sensor connection...")
        connection, client_address = secure_socket.accept()
        client_thread = threading.Thread(target=handle_client, args=(connection, client_address))
        client_thread.start()

if __name__ == "__main__":
    main()
