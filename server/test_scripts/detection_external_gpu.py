from ultralytics import YOLO
import torch
import cv2
import cvzone

# Load the YOLO model
model = YOLO('yolov10n.pt')

# Move the model to the GPU
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
        boxes = result.boxes

        # Check if any bounding boxes were detected
        if boxes is not None:
            for box in boxes:
                # Extract bounding box coordinates and class information
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype('int')  # Convert to CPU for drawing
                confidence = box.conf[0].cpu().numpy() * 100  # Confidence as a percentage
                class_detected_number = int(box.cls[0])
                class_detected_name = result.names[class_detected_number]

                # Draw bounding box and label on the image
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cvzone.putTextRect(image, f'{class_detected_name} {confidence:.1f}%', 
                                   [x1 + 8, y1 - 12], thickness=2, scale=1.5)

    # Display the image with detections
    cv2.imshow('frame', image)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
