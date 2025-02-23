from ultralytics import YOLO
import torch
import cv2
import cvzone
import numpy as np
import mediapipe as mp

# Load the YOLO pose model
model = YOLO('yolov8n-pose.pt')

# Move the model to the GPU if available
if torch.cuda.is_available():
    model.to('cuda')
else:
    print("CUDA not available. Using CPU.")

# Initialize the external USB camera (index may need to be adjusted)
cap = cv2.VideoCapture(0)  # Try index 1 for an external camera, adjust if necessary

# Check if the external camera was successfully opened
if not cap.isOpened():
    print("Error: Could not open external camera.")
    exit()

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize MediaPipe Pose for head orientation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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

def calculate_head_roll(face_landmarks, image_width, image_height):
    # Get coordinates of left and right eye outer corners
    left_eye_outer = face_landmarks.landmark[33]  # Left eye outer corner
    right_eye_outer = face_landmarks.landmark[263]  # Right eye outer corner

    x1 = left_eye_outer.x * image_width
    y1 = left_eye_outer.y * image_height
    x2 = right_eye_outer.x * image_width
    y2 = right_eye_outer.y * image_height

    # Calculate angle in radians
    angle_radians = np.arctan2(y2 - y1, x2 - x1)
    # Convert to degrees
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

def determine_pose(keypoints_xy, bbox, head_roll_angle=None):
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

    # Helper function to check if a keypoint is valid
    def is_valid(kp):
        return not np.array_equal(kp, [0, 0])

    votes = {'Standing': 0, 'Sitting': 0, 'Lying Down': 0}

    # Initialize weights for the voting system
    WEIGHT_HEAD_POSITION = 2
    WEIGHT_HEAD_ORIENTATION = 3
    WEIGHT_BODY_ALIGNMENT = 2
    WEIGHT_ARM_POSITION = 1
    WEIGHT_LEG_ANGLES = 1
    WEIGHT_BBOX_ASPECT = 1

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
    if head_roll_angle is not None:
        if abs(head_roll_angle) < 20:
            # Head is upright
            votes['Standing'] += WEIGHT_HEAD_ORIENTATION
            votes['Sitting'] += WEIGHT_HEAD_ORIENTATION
        elif abs(head_roll_angle) > 45:
            # Head is sideways
            votes['Lying Down'] += WEIGHT_HEAD_ORIENTATION
        else:
            # Head is tilted
            votes['Sitting'] += WEIGHT_HEAD_ORIENTATION

    # Rule 3: Body Alignment (Vertical)
    keypoints_y = []
    keypoints_indices = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]
    for idx in keypoints_indices:
        if is_valid(keypoints_xy[idx]):
            keypoints_y.append(keypoints_xy[idx][1])
    if keypoints_y:
        y_range = np.max(keypoints_y) - np.min(keypoints_y)
        if y_range > 150:  # Adjust threshold based on image size
            votes['Standing'] += WEIGHT_BODY_ALIGNMENT
            votes['Sitting'] += WEIGHT_BODY_ALIGNMENT
        else:
            votes['Lying Down'] += WEIGHT_BODY_ALIGNMENT

    # Rule 4: Arm Position Relative to Body
    # If wrists are near the hips or shoulders horizontally
    if is_valid(keypoints_xy[LEFT_WRIST]) and is_valid(keypoints_xy[LEFT_HIP]):
        wrist_hip_distance = abs(keypoints_xy[LEFT_WRIST][0] - keypoints_xy[LEFT_HIP][0])
        if wrist_hip_distance < 50:  # Threshold in pixels
            votes['Standing'] += WEIGHT_ARM_POSITION
            votes['Sitting'] += WEIGHT_ARM_POSITION
        else:
            votes['Lying Down'] += WEIGHT_ARM_POSITION

    if is_valid(keypoints_xy[RIGHT_WRIST]) and is_valid(keypoints_xy[RIGHT_HIP]):
        wrist_hip_distance = abs(keypoints_xy[RIGHT_WRIST][0] - keypoints_xy[RIGHT_HIP][0])
        if wrist_hip_distance < 50:
            votes['Standing'] += WEIGHT_ARM_POSITION
            votes['Sitting'] += WEIGHT_ARM_POSITION
        else:
            votes['Lying Down'] += WEIGHT_ARM_POSITION

    # Rule 5: Leg Angles (Knee angles)
    def calculate_angle(a, b, c):
        # Calculate the angle at point b given three points a-b-c
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        return angle

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
            votes['Sitting'] += WEIGHT_LEG_ANGLES
        else:
            votes['Standing'] += WEIGHT_LEG_ANGLES
            votes['Lying Down'] += WEIGHT_LEG_ANGLES

    # Rule 6: Bounding Box Aspect Ratio
    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    if bbox_height > bbox_width:
        votes['Standing'] += WEIGHT_BBOX_ASPECT
        votes['Sitting'] += WEIGHT_BBOX_ASPECT
    else:
        votes['Lying Down'] += WEIGHT_BBOX_ASPECT

    # Rule 7: Only Head Visible
    # If bounding box is small and only head keypoints are valid
    valid_keypoints = [idx for idx in range(len(keypoints_xy)) if is_valid(keypoints_xy[idx])]
    if len(valid_keypoints) <= 5:  # Only a few keypoints detected
        if is_valid(keypoints_xy[NOSE]):
            if head_roll_angle is not None and abs(head_roll_angle) < 20:
                votes['Standing'] += WEIGHT_HEAD_POSITION
                votes['Sitting'] += WEIGHT_HEAD_POSITION
            elif head_roll_angle is not None and abs(head_roll_angle) > 45:
                votes['Lying Down'] += WEIGHT_HEAD_POSITION
            else:
                votes['Sitting'] += WEIGHT_HEAD_POSITION

    # Determine the pose based on votes
    max_votes = max(votes.values())
    possible_poses = [pose for pose, vote in votes.items() if vote == max_votes]

    # Decide the final pose
    if 'Standing' in possible_poses and 'Sitting' in possible_poses:
        # Use additional cues to differentiate
        if 'Sitting' in votes and votes['Sitting'] > votes['Standing']:
            final_pose = 'Sitting'
        else:
            final_pose = 'Standing'
    elif len(possible_poses) == 1:
        final_pose = possible_poses[0]
    else:
        final_pose = 'Pose Unknown'

    return final_pose

# Start processing the camera feed
while True:
    ret, image = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Convert the image color for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform detection on the captured frame
    results = model(image)

    # Loop through detected results
    for result in results:
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

                # Initialize head roll angle
                head_roll_angle = None

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

                        # Calculate head roll angle
                        head_roll_angle = calculate_head_roll(face_landmarks, w, h)

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
                pose_label = determine_pose(keypoints_xy, (x1, y1, x2, y2), head_roll_angle)

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
