from ultralytics import YOLO
import torch
import cv2
import cvzone
import numpy as np

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

def get_angle_with_vertical(vector):
    # Negative y because image coordinates y increases downward
    vertical_vector = np.array([0, -1])
    unit_vector = vector / (np.linalg.norm(vector) + 1e-6)
    unit_vertical = vertical_vector / (np.linalg.norm(vertical_vector) + 1e-6)
    dot_product = np.dot(unit_vector, unit_vertical)
    # Ensure the dot product is within the valid range
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle = np.arccos(dot_product)
    angle_degrees = np.degrees(angle)
    return angle_degrees

def determine_pose(keypoints_xy):
    required_indices = [5, 6, 11, 12]  # Indices for left/right shoulders and hips
    num_keypoints = keypoints_xy.shape[0]

    # Check if all required keypoints are available
    if num_keypoints <= max(required_indices):
        return 'Pose Unknown'

    # Extract relevant keypoints
    left_shoulder = keypoints_xy[5]
    right_shoulder = keypoints_xy[6]
    left_hip = keypoints_xy[11]
    right_hip = keypoints_xy[12]

    # Check if keypoints are detected (keypoints might be [0, 0] if not detected)
    if np.any(left_shoulder == [0, 0]) or np.any(right_shoulder == [0, 0]) or \
       np.any(left_hip == [0, 0]) or np.any(right_hip == [0, 0]):
        return 'Pose Unknown'

    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2

    body_vector = hip_center - shoulder_center

    angle_degrees = get_angle_with_vertical(body_vector)

    if angle_degrees < 30:
        return 'Standing'
    elif angle_degrees > 60:
        return 'Lying Down'
    else:
        return 'Sitting'

# Start processing the camera feed
while True:
    ret, image = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        print("Error: Failed to capture image.")
        break

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
                # For debugging, you can print the shape:
                # print('keypoints_xy shape:', keypoints_xy.shape)

                # Determine pose
                pose_label = determine_pose(keypoints_xy)

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
