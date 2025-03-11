import numpy as np
import time
import cv2
# Constants for pose determination
HEAD_PITCH_THRESHOLD = 15  # Degrees
HEAD_ROLL_THRESHOLD = 15  # Degrees
HEAD_SIDEWAYS_THRESHOLD = 45  # Degrees
ASPECT_RATIO_THRESHOLD = 1.5
MIN_BBOX_HEIGHT = 100
Y_RANGE_THRESHOLD = 150  # Pixels
POSE_CONFIDENCE_THRESHOLD = 0.1  # 10% difference
# Weights for the voting system
WEIGHT_HEAD_POSITION = 2
WEIGHT_HEAD_ORIENTATION = 3
WEIGHT_BODY_ALIGNMENT = 2
WEIGHT_ARM_POSITION = 1
WEIGHT_LEG_ANGLES = 1
WEIGHT_BBOX_ASPECT = 1
WEIGHT_OBJECT_NEARBY = 2

class PersonTracker:
    def __init__(self):
        self.tracks = {}  # track_id: track_info
        self.next_id = 1
        self.max_age = 5  # frames
        self.max_distance = 50  # pixels
        self.determination_time = 3  # seconds
    
    def bbox_iou(self, bbox1, bbox2):
        """Calculate IoU (Intersection over Union) between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0  # No intersection
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IoU
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    def bbox_overlaps(self, bbox1, bbox2, threshold=0.2):
        """Check if two bounding boxes overlap significantly."""
        iou = self.bbox_iou(bbox1, bbox2)
        return iou > threshold
        
    def merge_overlapping_detections(self, detections):
        """Merge detections that likely belong to the same person."""
        if not detections:
            return []
            
        # Sort detections by area (largest first)
        detections = sorted(detections, key=lambda d: 
                          (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1]), 
                          reverse=True)
        
        merged_detections = []
        used_indices = set()
        
        for i, detection in enumerate(detections):
            if i in used_indices:
                continue
                
            merged_bbox = list(detection['bbox'])
            merged_centroid = list(detection['centroid'])
            eye_status = detection['eye_status']
            pose = detection['pose']
            overlap_count = 1
            used_indices.add(i)
            
            # Check for overlaps with other detections
            for j, other in enumerate(detections):
                if j in used_indices or j == i:
                    continue
                    
                if self.bbox_overlaps(detection['bbox'], other['bbox'], threshold=0.15):
                    # Merge bounding boxes
                    merged_bbox[0] = min(merged_bbox[0], other['bbox'][0])  # x1
                    merged_bbox[1] = min(merged_bbox[1], other['bbox'][1])  # y1
                    merged_bbox[2] = max(merged_bbox[2], other['bbox'][2])  # x2
                    merged_bbox[3] = max(merged_bbox[3], other['bbox'][3])  # y2
                    
                    # Update centroid
                    merged_centroid[0] += other['centroid'][0]
                    merged_centroid[1] += other['centroid'][1]
                    overlap_count += 1
                    
                    # Prefer "Eyes Closed" status for better safety
                    if other['eye_status'] == "Eyes Closed":
                        eye_status = "Eyes Closed"
                        
                    # Prefer sitting/lying down poses for better safety
                    if other['pose'] in ["Sitting", "Lying Down"]:
                        pose = other['pose']
                        
                    used_indices.add(j)
            
            # Calculate average centroid
            if overlap_count > 1:
                merged_centroid[0] /= overlap_count
                merged_centroid[1] /= overlap_count
            
            merged_detection = {
                'bbox': tuple(merged_bbox),
                'centroid': tuple(merged_centroid),
                'eye_status': eye_status,
                'pose': pose
            }
            merged_detections.append(merged_detection)
            
        return merged_detections
    
    def update(self, detections):
        # First, merge overlapping detections to avoid duplicate tracks
        merged_detections = self.merge_overlapping_detections(detections)
        
        updated_tracks = {}
        for detection in merged_detections:
            detection_centroid = detection['centroid']
            detection_bbox = detection['bbox']
            min_distance = float('inf')
            matched_track_id = None
            
            # Match to existing tracks - try both centroid distance and IoU
            for track_id, track in self.tracks.items():
                # Check centroid distance
                track_centroid = track['centroid']
                distance = np.linalg.norm(np.array(detection_centroid) - np.array(track_centroid))
                
                # Check bounding box overlap
                iou = self.bbox_iou(track['bbox'], detection_bbox)
                
                # Use a weighted score considering both distance and IoU
                # Lower distance and higher IoU is better
                weighted_score = distance / (1 + 10 * iou)  # IoU has higher weight
                
                # Increase max_distance for larger boxes (people closer to camera)
                bbox_size = (detection_bbox[2] - detection_bbox[0]) * (detection_bbox[3] - detection_bbox[1])
                adaptive_max_distance = self.max_distance * (1 + (bbox_size / 40000))  # Adjust threshold based on size
                
                if weighted_score < min_distance and (distance < adaptive_max_distance or iou > 0.25):
                    min_distance = weighted_score
                    matched_track_id = track_id
                    
            if matched_track_id is not None:
                # Update existing track
                track = self.tracks[matched_track_id]
                track['bbox'] = detection['bbox']
                track['centroid'] = detection['centroid']
                track['age'] = 0
                
                # Add new data
                timestamp = time.time()
                track['data'].append({
                    'timestamp': timestamp,
                    'pose': detection['pose'],
                    'eye_status': detection['eye_status']
                })
                
                # Trim old data
                self.trim_data(track)
                updated_tracks[matched_track_id] = track
            else:
                # Create new track
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
        
        # Age unmatched tracks
        for track_id in self.tracks.keys():
            if track_id not in updated_tracks:
                track = self.tracks[track_id]
                track['age'] += 1
                if track['age'] <= self.max_age:
                    updated_tracks[track_id] = track
        
        # Update tracks dictionary - THIS LINE WAS MISSING
        self.tracks = updated_tracks

    def trim_data(self, track):
        current_time = time.time()
        # Keep only data from the last X seconds for unconsciousness check
        track['data'] = [entry for entry in track['data'] 
                         if current_time - entry['timestamp'] <= self.determination_time]
    
    def check_unconscious(self, track):
        data = track['data']
        if not data:
            return False
            
        # Determine if unconscious criteria are met
        poses = [entry['pose'] for entry in data]
        eye_statuses = [entry['eye_status'] for entry in data]
        
        sitting_or_lying = [pose in ['Sitting', 'Lying Down'] for pose in poses]
        eyes_closed = [eye_status == 'Eyes Closed' for eye_status in eye_statuses]
        
        percent_sitting_or_lying = sum(sitting_or_lying) / len(data)
        percent_eyes_closed = sum(eyes_closed) / len(data)
        
        # Criteria to determine unconsciousness
        unconscious_criteria_met = percent_sitting_or_lying >= 0.8 and percent_eyes_closed >= 0.8
        
        current_time = time.time()
        
        if unconscious_criteria_met:
            # If criteria met, set unconscious status and timestamp if not already set
            if not track.get('unconscious'):
                track['unconscious'] = True
                track['unconscious_since'] = current_time
        else:
            # Reset unconscious status
            track['unconscious'] = False
            track['unconscious_since'] = None
            
        return track['unconscious']

# Helper functions for pose determination
def is_valid(kp):
    """Check if a keypoint is valid (non-zero)"""
    return not np.array_equal(kp, [0, 0])

def calculate_angle(a, b, c):
    """Calculate angle at point b given three points a-b-c"""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle

def calculate_EAR(landmarks, eye_indices):
    """Calculate Eye Aspect Ratio"""
    p2_minus_p6 = np.linalg.norm(landmarks[eye_indices[1]] - landmarks[eye_indices[5]])
    p3_minus_p5 = np.linalg.norm(landmarks[eye_indices[2]] - landmarks[eye_indices[4]])
    p1_minus_p4 = np.linalg.norm(landmarks[eye_indices[0]] - landmarks[eye_indices[3]])
    return (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4 + 1e-6)

def calculate_head_angles(face_landmarks, image_width, image_height):
    """Calculate head roll and pitch angles from face landmarks"""
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
    """Determine person's pose using a voting system"""
    # Define indices for keypoints
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
    
    # Check if we can see below the torso (hips/legs)
    lower_body_keypoints = [LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]
    visible_lower_body = sum(1 for kp in lower_body_keypoints if is_valid(keypoints_xy[kp]))
    
    # If we can't see enough of the lower body, assume sitting
    # We need at least hips and some leg keypoints to be confident
    if visible_lower_body < 2:
        # We can still see torso, but not much below
        if is_valid(keypoints_xy[LEFT_SHOULDER]) or is_valid(keypoints_xy[RIGHT_SHOULDER]):
            return 'Sitting'
    
    votes = {'Standing': 0, 'Sitting': 0, 'Lying Down': 0}
    
    # Rule 1: Head Position Relative to Shoulders
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
                # Head is above shoulders
                votes['Standing'] += WEIGHT_HEAD_POSITION
                votes['Sitting'] += WEIGHT_HEAD_POSITION
            else:
                # Head not above shoulders
                votes['Lying Down'] += WEIGHT_HEAD_POSITION
    
    # Rule 2: Head Orientation
    if head_angles is not None:
        head_roll_angle, head_pitch_angle = head_angles
        if abs(head_roll_angle) < HEAD_ROLL_THRESHOLD and abs(head_pitch_angle) < HEAD_PITCH_THRESHOLD:
            # Head is upright
            votes['Standing'] += WEIGHT_HEAD_ORIENTATION
            votes['Sitting'] += WEIGHT_HEAD_ORIENTATION
        elif abs(head_roll_angle) > HEAD_SIDEWAYS_THRESHOLD or abs(head_pitch_angle) > HEAD_SIDEWAYS_THRESHOLD:
            # Head is significantly tilted or sideways
            votes['Lying Down'] += WEIGHT_HEAD_ORIENTATION
        else:
            # Head is slightly tilted
            votes['Sitting'] += WEIGHT_HEAD_ORIENTATION
    
    # Rule 3: Body Alignment (Vertical)
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
    
    # Rule 4: Arm Position Relative to Body
    elbow_angles = []
    if is_valid(keypoints_xy[LEFT_SHOULDER]) and is_valid(keypoints_xy[LEFT_ELBOW]) and is_valid(keypoints_xy[LEFT_WRIST]):
        angle = calculate_angle(keypoints_xy[LEFT_SHOULDER], keypoints_xy[LEFT_ELBOW], keypoints_xy[LEFT_WRIST])
        elbow_angles.append(angle)
    if is_valid(keypoints_xy[RIGHT_SHOULDER]) and is_valid(keypoints_xy[RIGHT_ELBOW]) and is_valid(keypoints_xy[RIGHT_WRIST]):
        angle = calculate_angle(keypoints_xy[RIGHT_SHOULDER], keypoints_xy[RIGHT_ELBOW], keypoints_xy[RIGHT_WRIST])
        elbow_angles.append(angle)
    
    if elbow_angles:
        avg_elbow_angle = np.mean(elbow_angles)
        if avg_elbow_angle < 160:  # Arms are bent
            votes['Sitting'] += WEIGHT_ARM_POSITION
        else:
            votes['Standing'] += WEIGHT_ARM_POSITION
            votes['Lying Down'] += WEIGHT_ARM_POSITION
    
    # Rule 5: Leg Angles (Knee angles)
    knee_angles = []
    if is_valid(keypoints_xy[LEFT_HIP]) and is_valid(keypoints_xy[LEFT_KNEE]) and is_valid(keypoints_xy[LEFT_ANKLE]):
        angle = calculate_angle(keypoints_xy[LEFT_HIP], keypoints_xy[LEFT_KNEE], keypoints_xy[LEFT_ANKLE])
        knee_angles.append(angle)
    if is_valid(keypoints_xy[RIGHT_HIP]) and is_valid(keypoints_xy[RIGHT_KNEE]) and is_valid(keypoints_xy[RIGHT_ANKLE]):
        angle = calculate_angle(keypoints_xy[RIGHT_HIP], keypoints_xy[RIGHT_KNEE], keypoints_xy[RIGHT_ANKLE])
        knee_angles.append(angle)
    
    if knee_angles:
        avg_knee_angle = np.mean(knee_angles)
        if avg_knee_angle < 120:  # Bent knees
            votes['Sitting'] += WEIGHT_LEG_ANGLES * 2  # Strong indicator
        else:
            votes['Standing'] += WEIGHT_LEG_ANGLES
            votes['Lying Down'] += WEIGHT_LEG_ANGLES
    
    # Rule 6: Bounding Box Aspect Ratio
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
    
    # Rule 7: Only Head Visible
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
    
    # Rule 8: Object Detection Influence
    person_bottom_y = max(
        y2,
        keypoints_xy[LEFT_HIP][1] if is_valid(keypoints_xy[LEFT_HIP]) else 0,
        keypoints_xy[RIGHT_HIP][1] if is_valid(keypoints_xy[RIGHT_HIP]) else 0,
        keypoints_xy[LEFT_KNEE][1] if is_valid(keypoints_xy[LEFT_KNEE]) else 0,
        keypoints_xy[RIGHT_KNEE][1] if is_valid(keypoints_xy[RIGHT_KNEE]) else 0
    )
    
    for obj in objects_detected:
        if obj['name'] in ['chair', 'sofa', 'bench']:
            # Check if the object is near the person's hips or knees
            chair_bbox = obj['bbox']
            chair_top_y = chair_bbox[1]
            if abs(chair_top_y - person_bottom_y) < 100:  # Threshold in pixels
                votes['Sitting'] += WEIGHT_OBJECT_NEARBY * 2  # Strong indicator
    
    # Rule 9: Feet Position
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
            if abs(avg_ankle_y - avg_hip_y) < 50:  # Threshold in pixels
                votes['Sitting'] += WEIGHT_BODY_ALIGNMENT
                votes['Lying Down'] += WEIGHT_BODY_ALIGNMENT
    
    # Determine the pose based on votes
    total_votes = sum(votes.values())
    if total_votes == 0:
        # If no votes, default to sitting if we can see upper body
        if is_valid(keypoints_xy[LEFT_SHOULDER]) or is_valid(keypoints_xy[RIGHT_SHOULDER]):
            return 'Sitting'
        return 'unknown'
    
    sorted_votes = sorted(votes.items(), key=lambda item: item[1], reverse=True)
    max_votes = sorted_votes[0][1]
    second_max_votes = sorted_votes[1][1]
    
    # Calculate percentage difference
    percentage_diff = (max_votes - second_max_votes) / total_votes if total_votes > 0 else 0
    
    # Decide the final pose
    if percentage_diff <= POSE_CONFIDENCE_THRESHOLD:
        # If the difference is small, return the most likely pose with a lower confidence
        return sorted_votes[0][0]
    else:
        return sorted_votes[0][0]