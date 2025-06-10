from ultralytics import YOLO
import cv2
model = YOLO('yolov8n.pt')
# Path to your downloaded video
video_path = r"C:\Users\Asus\Downloads\TEST 3.mp4"

# Load the video
video = cv2.VideoCapture(video_path)

# Check if the video loaded successfully
if not video.isOpened():
    print("Error: Could not open video.")
    exit()

# Read and display each frame from the video
while True:
    ret, frame = video.read()
    
    
    # If video is over, break
    if not ret:
        print("Video finished.")
        break
    
    frame = cv2.resize(frame, (640, 480))
    results=model.track(frame, persist=True)

    frame_ = results[0].plot()
    # Display the current frame
    cv2.imshow('Video Playback', frame_)

    # Press 'q' to stop the video playback
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video object and close all windows
video.release()
cv2.destroyAllWindows()
