import cv2
import numpy as np
import torch
from ultralytics import YOLO
import cvzone
import mediapipe as mp

import logging

# Set logging level to WARNING to suppress INFO logs
logging.getLogger('ultralytics').setLevel(logging.WARNING)

# ==============================
# Configuration Variables
# ==============================

# Threshold for pose ambiguity (percentage difference between top two poses)
POSE_CONFIDENCE_THRESHOLD = 0.1 # 10% difference

HEAD_PITCH_THRESHOLD = 15  # Degrees
HEAD_ROLL_THRESHOLD = 15  # Degrees
HEAD_SIDEWAYS_THRESHOLD = 45  # Degrees
ASPECT_RATIO_THRESHOLD = 1.5  # Adjust as needed
MIN_BBOX_HEIGHT = 100  # Minimum height to consider the aspect ratio rule

UNCONSCIOUSNESS_PROBABILITY = 0.3

# Weights for the voting system
WEIGHT_HEAD_POSITION = 2
WEIGHT_HEAD_ORIENTATION = 3
WEIGHT_BODY_ALIGNMENT = 2
WEIGHT_ARM_POSITION = 1
WEIGHT_LEG_ANGLES = 1
WEIGHT_BBOX_ASPECT = 1
WEIGHT_OBJECT_NEARBY = 2  # Weight for object detection (e.g., chair)

# Thresholds for angles and distances
HEAD_ROLL_THRESHOLD = 20  # Degrees
HEAD_SIDEWAYS_THRESHOLD = 45  # Degrees
ARM_DISTANCE_THRESHOLD = 50  # Pixels
KNEE_ANGLE_BENT = 120  # Degrees
Y_RANGE_THRESHOLD = 150  # Pixels
VOTE_DIFF_THRESHOLD = 0  # Set to 0 for exact tie, or adjust as needed

# ==============================
# Load Models
# ==============================

# Load the YOLO pose model
pose_model = YOLO('yolov8n-pose.pt', verbose=False)  # Pose estimation model

# Load the YOLO object detection model
object_model = YOLO('yolov8n.pt', verbose=False)  # Object detection model (trained on COCO dataset)

# Move models to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pose_model.to(device)
object_model.to(device)

# ==============================
# Initialize Camera
# ==============================

# Initialize the external USB camera (index may need to be adjusted)
cap = cv2.VideoCapture(0)  # Adjust the index if necessary

# Check if the external camera was successfully opened
if not cap.isOpened():
    print("Error: Could not open external camera.")
    exit()

# ==============================
# Initialize MediaPipe Face Mesh
# ==============================

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==============================
# Helper Functions
# ==============================

def calculate_EAR(landmarks, eye_indices):
    # Compute the euclidean distances between the vertical eye landmarks
    p2_minus_p6 = np.linalg.norm(landmarks[eye_indices[1]] - landmarks[eye_indices[5]])
    p3_minus_p5 = np.linalg.norm(landmarks[eye_indices[2]] - landmarks[eye_indices[4]])

    # Compute the euclidean distance between the horizontal eye landmarks
    p1_minus_p4 = np.linalg.norm(landmarks[eye_indices[0]] - landmarks[eye_indices[3]])

    # Compute EAR
    ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4 + 1e-6)
    return ear

# Eye landmarks indices based on MediaPipe Face Mesh
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.25  # Threshold for eye closure

def calculate_head_angles(face_landmarks, image_width, image_height):
    # Calculate head roll angle (unchanged)
    left_eye_outer = face_landmarks.landmark[33]
    right_eye_outer = face_landmarks.landmark[263]

    x1 = left_eye_outer.x * image_width
    y1 = left_eye_outer.y * image_height
    x2 = right_eye_outer.x * image_width
    y2 = right_eye_outer.y * image_height

    # Corrected roll angle calculation
    angle_radians_roll = np.arctan2(y1 - y2, x2 - x1)
    angle_degrees_roll = np.degrees(angle_radians_roll)

    # Calculate head pitch using nose tip and chin
    nose_tip = face_landmarks.landmark[1]
    chin = face_landmarks.landmark[152]

    x_nose = nose_tip.x * image_width
    y_nose = nose_tip.y * image_height
    x_chin = chin.x * image_width
    y_chin = chin.y * image_height

    # Corrected pitch angle calculation
    angle_radians_pitch = np.arctan2(y_chin - y_nose, x_chin - x_nose)
    angle_degrees_pitch = np.degrees(angle_radians_pitch)

    # Adjust the pitch angle to have 0 degrees when the head is upright
    # and positive when looking down, negative when looking up
    pitch_angle_adjusted = angle_degrees_pitch - 90

    return angle_degrees_roll, pitch_angle_adjusted

def is_valid(kp):
    return not np.array_equal(kp, [0, 0])

def calculate_angle(a, b, c):
    # Calculate the angle at point b given three points a-b-c
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle

def determine_pose(keypoints_xy, bbox, head_angles=None, objects_detected=[]):
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

    # Adjusted head orientation rule
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
    else:
        # Not enough keypoints to make a reliable decision
        pass




    # Rule 4: Arm Position Relative to Body
    # Analyze angles at elbows
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
        if avg_knee_angle < KNEE_ANGLE_BENT:  # Bent knees
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
    else:
        # Bounding box too small; skip this rule
        pass


    # Rule 7: Only Head Visible
    # If bounding box is small and only head keypoints are valid
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
    # Check if a chair is detected near the lower part of the person's body
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
    # If ankles are at similar height as hips, might be sitting or lying down
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
                votes['Sitting'] += WEIGHT_BODY_ALIGNMENT * 2
                votes['Lying Down'] += WEIGHT_BODY_ALIGNMENT * 2

    # Determine the pose based on votes
    total_votes = sum(votes.values())
    if total_votes == 0:
        return 'Pose Unknown'

    sorted_votes = sorted(votes.items(), key=lambda item: item[1], reverse=True)
    max_votes = sorted_votes[0][1]
    second_max_votes = sorted_votes[1][1]

    # Calculate percentage difference
    percentage_diff = (max_votes - second_max_votes) / total_votes

    # print (votes)
    # Decide the final pose
    if percentage_diff <= POSE_CONFIDENCE_THRESHOLD:
        # If the difference is small, output multiple possible poses
        possible_poses = [pose for pose, vote in votes.items() if vote == max_votes or vote == second_max_votes]
        final_pose = ' or '.join(possible_poses)
    else:
        final_pose = sorted_votes[0][0]

    return final_pose

# ==============================
# Main Loop
# ==============================

while True:
    ret, image = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Convert the image color for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform object detection on the captured frame
    object_results = object_model(image)

    # Perform pose estimation on the captured frame
    pose_results = pose_model(image)

    # Extract object detections
    objects_detected = []
    for result in object_results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1_obj, y1_obj, x2_obj, y2_obj = box.xyxy[0].cpu().numpy().astype('int')
                confidence_obj = box.conf[0].cpu().numpy() * 100  # Confidence as a percentage
                class_detected_number_obj = int(box.cls[0])
                class_detected_name_obj = result.names[class_detected_number_obj]
                objects_detected.append({
                    'name': class_detected_name_obj,
                    'bbox': (x1_obj, y1_obj, x2_obj, y2_obj),
                    'confidence': confidence_obj
                })
                # Draw bounding box and label on the image
                cv2.rectangle(image, (x1_obj, y1_obj), (x2_obj, y2_obj), (255, 0, 0), 2)
                cvzone.putTextRect(
                    image,
                    f'{class_detected_name_obj} {confidence_obj:.1f}%',
                    [x1_obj + 5, y1_obj - 10],
                    thickness=1,
                    scale=1,
                    colorR=(255, 0, 0)
                )

    # Loop through pose estimation results
    for result in pose_results:
        boxes = result.boxes  # Bounding boxes
        keypoints_list = result.keypoints  # Pose keypoints

        # Check if any detections were made
        if boxes is not None and keypoints_list is not None:
            for idx in range(len(boxes)):
                box = boxes[idx]
                keypoint = keypoints_list[idx]

                # Extract bounding box coordinates and class information
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype('int')
                confidence = box.conf[0].cpu().numpy() * 100  # Confidence as a percentage
                class_detected_number = int(box.cls[0])
                class_detected_name = result.names[class_detected_number]

                # Draw bounding box and label on the image
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cvzone.putTextRect(
                    image,
                    f'{class_detected_name} {confidence:.1f}%',
                    [x1 + 8, y1 - 12],
                    thickness=2,
                    scale=1.5
                )

                # Extract keypoints as numpy array
                keypoints_xy = keypoint.xy.cpu().numpy().squeeze()  # Shape: (num_keypoints, 2)

                # Initialize head angles
                head_angles = None

                # Extract face region for MediaPipe
                face_region = image_rgb[y1:y2, x1:x2]

                # Process the face region with MediaPipe Face Mesh
                face_results = face_mesh.process(face_region)

                eye_status = "Eyes Open"

                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        # Convert normalized landmarks to pixel coordinates
                        h, w, _ = face_region.shape
                        landmarks = []
                        for lm in face_landmarks.landmark:
                            x, y = int(lm.x * w), int(lm.y * h)
                            landmarks.append([x, y])
                        landmarks = np.array(landmarks)

                        # Calculate EAR for both eyes
                        left_ear = calculate_EAR(landmarks, LEFT_EYE_INDICES)
                        right_ear = calculate_EAR(landmarks, RIGHT_EYE_INDICES)
                        avg_ear = (left_ear + right_ear) / 2.0

                        # Determine if eyes are closed
                        if avg_ear < EAR_THRESHOLD:
                            eye_status = "Eyes Closed"

                        # Calculate head angles
                        head_roll_angle, head_pitch_angle = calculate_head_angles(face_landmarks, w, h)
                        head_angles = (head_roll_angle, head_pitch_angle)

                        # Draw eye status on the image
                        cvzone.putTextRect(
                            image,
                            f'Eye Status: {eye_status}',
                            [x1 + 8, y2 + 70],
                            thickness=2,
                            scale=1.5
                        )

                        # Print eye status to the console
                        print(f"Person {idx + 1}: {eye_status}")

                        # Break after first face (since we set max_num_faces=1)
                        break
                else:
                    # If no face landmarks detected
                    cvzone.putTextRect(
                        image,
                        f'Eye Status: Unknown',
                        [x1 + 8, y2 + 70],
                        thickness=2,
                        scale=1.5
                    )
                    print(f"Person {idx + 1}: Eye Status Unknown")

                # Determine pose
                pose_label = determine_pose(keypoints_xy, (x1, y1, x2, y2), head_angles, objects_detected)

                # Print pose estimation to the console
                print(f"Person {idx + 1}: Pose - {pose_label}")

                # Draw pose label
                cvzone.putTextRect(
                    image,
                    f'Pose: {pose_label}',
                    [x1 + 8, y2 + 30],
                    thickness=2,
                    scale=1.5
                )

                # Optionally draw keypoints
                for kp in keypoints_xy:
                    x_kp, y_kp = int(kp[0]), int(kp[1])
                    cv2.circle(image, (x_kp, y_kp), 3, (0, 255, 0), -1)

    # Display the image with detections and pose estimations
    cv2.imshow('frame', image)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
