import cv2
import face_recognition
import numpy as np
import csv
import pandas as pd
from ultralytics import YOLO
import easyocr
from datetime import datetime
import os


class FaceRecognizer:
    def __init__(self, csv_file, tolerance=0.4, unknown_threshold=0.6):
        self.known_faces = []
        self.known_names = []
        self.tolerance = tolerance
        self.unknown_threshold = unknown_threshold
        self.load_faces_from_csv(csv_file)


    def load_faces_from_csv(self, csv_file):
        if not os.path.exists(csv_file):
            print(f"Face encodings CSV file not found: {csv_file}")
            return
       
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) < 2:
                    continue
                name = row[0]
                encoding = np.array([float(x) for x in row[1:]])
                self.known_faces.append(encoding)
                self.known_names.append(name)
        print(f"Loaded {len(self.known_faces)} face encodings")


    def recognize_faces(self, frame):
        if frame is None or frame.size == 0:
            return []


        try:
            # Resize frame for faster processing, adjusting to 0.5 scaling for efficiency
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])  # Convert BGR to RGB
           
            # Detect faces and their encodings in the smaller frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
           
            recognized_faces = []
            for face_encoding, face_location in zip(face_encodings, face_locations):
                distances = face_recognition.face_distance(self.known_faces, face_encoding)
                best_match_index = np.argmin(distances)
                best_distance = distances[best_match_index]
                name = "Unknown"


                # Determine if the face is known based on distance thresholds
                if best_distance < self.tolerance:
                    name = self.known_names[best_match_index]
                elif best_distance > self.unknown_threshold:
                    name = "Unknown"


                # Adjust face location to original frame size
                top, right, bottom, left = face_location
                scale_factor = 2  # Since we resized by 0.5
                top, right, bottom, left = int(top * scale_factor), int(right * scale_factor), int(bottom * scale_factor), int(left * scale_factor)


                recognized_faces.append({
                    'name': name,
                    'confidence': 1 - best_distance,
                    'location': (top, right, bottom, left)
                })


            return recognized_faces


        except Exception as e:
            print(f"Error in face recognition: {str(e)}")
            return []


class LicensePlateSystem:
    def __init__(self, authorized_plates_path, model_path):
        if not os.path.exists(model_path) or not os.path.exists(authorized_plates_path):
            print("Error: Required files not found")
            return
               
        self.plate_detector = YOLO(model_path)
        self.reader = easyocr.Reader(['en'])
        self.authorized_plates = set(pd.read_csv(authorized_plates_path)['license_number'].str.upper().str.replace(' ', ''))
        self.detected_plates = set()
        print("License plate system initialized successfully")


    def read_license_plate(self, plate_crop):
        if plate_crop is None or plate_crop.size == 0:
            return None, None
           
        try:
            detections = self.reader.readtext(plate_crop)
            if detections:
                text = detections[0][-2].upper().replace(' ', '')
                confidence = detections[0][-1]
                if  len(text) == 10 and text.isalnum():
                    return text, confidence
            return None, None
        except Exception as e:
            print(f"Error reading license plate: {str(e)}")
            return None, None


    def detect_and_verify_plates(self, frame):
        if frame is None or frame.size == 0:
            return []
           
        try:
            results = self.plate_detector.predict(frame)
            unauthorized_plates = []
           
            for result in results:
                for box in result.boxes:
                    try:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        if x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
                            continue
                           
                        plate_crop = frame[y1:y2, x1:x2]
                        plate_text, confidence = self.read_license_plate(plate_crop)
                       
                        if plate_text:
                            if plate_text not in self.authorized_plates and plate_text not in self.detected_plates:
                                self.detected_plates.add(plate_text)
                                print(f"Unauthorized plate detected: {plate_text}")
                                unauthorized_plates.append({
                                    'text': plate_text,
                                    'confidence': confidence,
                                    'bbox': (x1, y1, x2, y2)
                                })
                               
                    except Exception as e:
                        print(f"Error processing plate: {str(e)}")
                        continue
                       
            return unauthorized_plates
           
        except Exception as e:
            print(f"Error in plate detection: {str(e)}")
            return []


def main():
    # Configuration
    config = {
        'face_encodings_path': r'C:\Users\Lenovo\Desktop\SecureNest\face_encodings.csv',
        'authorized_plates_path': r'C:\Users\Lenovo\Desktop\SecureNest\authorized_plates.csv',
        'license_plate_model_path': r'C:\Users\Lenovo\Desktop\SecureNest\best.pt',
        'video_path': r'C:\Users\Lenovo\Desktop\SecureNest\demo3.mp4',
        'output_path': r'C:\Users\Lenovo\Desktop\SecureNest\output_annotated_video3.mp4'
    }
   
    try:
        # Initialize components
        face_recognizer = FaceRecognizer(config['face_encodings_path'])
        license_plate_system = LicensePlateSystem(
            config['authorized_plates_path'],
            config['license_plate_model_path']
        )
        object_detector = YOLO('yolov8n.pt')
        detected_objects = set()
       
        # Open video source
        video_source = cv2.VideoCapture(config['video_path'])
        if not video_source.isOpened():
            print("Error: Could not open video source")
            return
       
        # Set up video writer
        frame_width = int(video_source.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_source.get(cv2.CAP_PROP_FPS))
       
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(config['output_path'], fourcc, fps, (frame_width, frame_height))
       
        print("Starting video processing...")
        frame_count = 0
        process_every_n_frames = 3
        while True:
            ret, frame = video_source.read()
            if not ret:
                break
               
            frame_count += 1
            if frame_count % process_every_n_frames != 0:
                continue
            print(f"Processing frame {frame_count}")
           
            # Process faces
            recognized_faces = face_recognizer.recognize_faces(frame)
            for face in recognized_faces:
                name = face['name']
                top, right, bottom, left = face['location']
                confidence = face['confidence']
               
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, f"{name} ({confidence:.2f})",
                           (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
           
            # Process license plates
            unauthorized_plates = license_plate_system.detect_and_verify_plates(frame)
            for plate in unauthorized_plates:
                x1, y1, x2, y2 = plate['bbox']
                text = plate['text']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"Unauthorized: {text}",
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
           
            # Process objects
            results = object_detector.predict(frame)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = result.names[cls]
                   
                    if label not in detected_objects and conf > 0.5:
                        detected_objects.add(label)
                        print(f"New object detected: {label}")
                   
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ({conf:.2f})",
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
           
            # Write the frame
            out.write(frame)
            cv2.imshow('Processed Video', frame)

            # Press 'q' to stop the video
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Video processing interrupted by user.")
                break
           
        print("Video processing completed")
       
    except Exception as e:
        print(f"Error: {str(e)}")
   
    finally:
        # Cleanup
        if 'video_source' in locals():
            video_source.release()
        if 'out' in locals():
            out.release()


if __name__ == "__main__":
    main()
