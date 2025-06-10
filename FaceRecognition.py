import cv2
import face_recognition
import numpy as np
import csv

class FaceRecognizer:
    def __init__(self, csv_file, tolerance=0.4, unknown_threshold=0.6):
        self.known_faces = []
        self.known_names = []
        self.tolerance = tolerance  # Reduced for stricter matching
        self.unknown_threshold = unknown_threshold
        self.load_faces_from_csv(csv_file)

    def load_faces_from_csv(self, csv_file):
        # Read encodings from CSV
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                name = row[0]  # The name is in the first column
                encoding = np.array([float(x) for x in row[1:]])  # The encoding follows the name
                self.known_faces.append(encoding)
                self.known_names.append(name)

    def recognize_faces(self, frame):
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Adjusted to 0.5 scaling for better quality
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])  # Convert BGR to RGB

        # Find faces in the resized frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        recognized_faces = []
        for face_encoding, face_location in zip(face_encodings, face_locations):
            distances = face_recognition.face_distance(self.known_faces, face_encoding)
            best_match_index = np.argmin(distances)
            best_distance = distances[best_match_index]
            name = "Unknown"

            # Apply stricter matching logic
            if best_distance < self.tolerance:
                name = self.known_names[best_match_index]
            elif best_distance > self.unknown_threshold:
                name = "Unknown"

            recognized_faces.append((name, face_location, best_distance))

        return recognized_faces

def main():
    csv_file = 'face_encodings.csv'  # Make sure this file exists
    video_path = 'C:/Users/asus/OneDrive/Desktop/Mini Project/Test Subjects/TEST 7.mp4'

    face_recognizer = FaceRecognizer(csv_file)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 2 == 0:  # Skip every second frame for performance
            recognized_faces = face_recognizer.recognize_faces(frame)

            for name, face_location, distance in recognized_faces:
                top, right, bottom, left = [i * 2 for i in face_location]  # Adjust back to original size
                if name == "Unknown":
                    color = (0, 0, 255)  # Red box for unknown faces
                else:
                    color = (0, 255, 0)  # Green box for known faces
                
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                label = f"{name} ({distance:.2f})" if name != "Unknown" else "Unknown"
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow("Video", frame)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
