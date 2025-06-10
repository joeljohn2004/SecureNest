import cv2
import face_recognition

# Load the video file
cap = cv2.VideoCapture('C:/Users/asus/OneDrive/Desktop/Mini Project/Test Subjects/Example.mp4')

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Loop through frames
while True:
    ret, frame = cap.read()

    # If the video has ended or cannot read the frame, break out of the loop
    if not ret:
        print("Reached the end of the video or cannot retrieve frame.")
        break

    # Convert the image from BGR (OpenCV format) to RGB (face_recognition format)
    rgb_frame = frame[:, :, ::-1]

    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)

    # Check if faces were detected
    if face_locations:
        print(f"Faces detected: {face_locations}")
    else:
        print("No faces detected.")

    # Draw a rectangle around each face
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Add a delay of 100ms and check if the 'Enter' key (ASCII code 13) was pressed
    if cv2.waitKey(500) & 0xFF == 13:
        break

# Release the video capture
cap.release()
