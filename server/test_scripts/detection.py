from ultralytics import YOLO
import cvzone
import cv2

# Load the YOLO model
model = YOLO('yolov10n.pt')

# Initialize the webcam (0 is the default camera index)
cap = cv2.VideoCapture(1)

# Check if the webcam was successfully opened
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Start processing the webcam feed
while True:
    ret, image = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Perform detection on the captured frame
    results = model(image)
    
    # Loop through detected results
    for info in results:
        boxes = info.boxes

        # Check if any bounding boxes were detected
        if boxes is not None:
            for box in boxes:
                # Extract bounding box coordinates and class information
                x1, y1, x2, y2 = box.xyxy[0].numpy().astype('int')
                confidence = box.conf[0].numpy() * 100  # Confidence as a percentage
                class_detected_number = int(box.cls[0])
                class_detected_name = info.names[class_detected_number]

                # Draw bounding box and label on the image
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cvzone.putTextRect(image, f'{class_detected_name} {confidence:.1f}%', 
                                   [x1 + 8, y1 - 12], thickness=2, scale=1.5)

    # Display the image with detections
    cv2.imshow('frame', image)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
