import cv2
import pickle
import numpy as np
import mysql.connector
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
import easyocr
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Database Initialization
def init_db():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="@Rida12348765",
        database="secure_nest_db"
    )
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS authorized_entries (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255),
            confidence FLOAT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS unauthorized_entries (
            id INT AUTO_INCREMENT PRIMARY KEY,
            entry_type VARCHAR(255),
            identifier VARCHAR(255),
            confidence FLOAT,
            face_encoding LONGBLOB,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    cursor.close()
    conn.close()

# Email Alert Function
def send_email_alert(entry_type, identifier, confidence):
    sender_email = "aaaaaaaa@gmail.com"
    receiver_email = "aaaaaaaaaa@rajagiri.edu.in"
    subject = f"Alert: New {entry_type} Entry"
    body = f"A new {entry_type} entry has been logged:\n\nIdentifier: {identifier}\nConfidence: {confidence}"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(sender_email, "password")
        server.sendmail(sender_email, receiver_email, msg.as_string())
    print("Email alert sent.")

# FaceRecognizer Class
class FaceRecognizer:
    def __init__(self, svm_model_path, embeddings_path, confidence_threshold=0.8, similarity_threshold=0.6):
        with open(svm_model_path, 'rb') as f:
            self.clf = pickle.load(f)
        data = np.load(embeddings_path)
        self.known_embeddings = data['embeddings']
        self.known_labels = data['labels']
        
        self.mtcnn = MTCNN(image_size=160, margin=10, keep_all=True, thresholds=[0.9, 0.98, 0.98])
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold
        self.tracked_faces = {}
        self.next_face_id = 0

    def recognize_faces(self, frame):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, _ = self.mtcnn.detect(img)
        recognized_faces = []

        if boxes is not None:
            faces = self.mtcnn(img)
            for face, box in zip(faces, boxes):
                x1, y1, x2, y2 = map(int, box)
                embedding = self.model(face.unsqueeze(0)).detach().numpy().flatten()
                
                label = self.clf.predict([embedding])[0]
                confidence = max(self.clf.predict_proba([embedding])[0])
                similarities = cosine_similarity([embedding], self.known_embeddings)
                max_similarity = float(np.max(similarities))

                if confidence < self.confidence_threshold or max_similarity < self.similarity_threshold:
                    label = "Unknown"

                face_id = self.track_face(embedding, label)
                if face_id in self.tracked_faces and self.tracked_faces[face_id]['label'] != "Unknown":
                    label = self.tracked_faces[face_id]['label']

                if 'logged' not in self.tracked_faces[face_id]:
                    self.log_face_entry(label, max_similarity, embedding)
                    self.tracked_faces[face_id]['logged'] = True

                # Store face details for annotation
                recognized_faces.append({
                    'name': label,
                    'confidence': max_similarity,
                    'location': (y1, x2, y2, x1)
                })

                self.tracked_faces[face_id] = {
                    'embedding': embedding,
                    'label': label,
                    'confidence': max_similarity,
                    'location': (y1, x2, y2, x1)
                }

        return recognized_faces

    def track_face(self, embedding, label):
        for face_id, face_data in self.tracked_faces.items():
            similarity = cosine_similarity([embedding], [face_data['embedding']])[0, 0]
            if similarity > self.similarity_threshold:
                return face_id

        self.tracked_faces[self.next_face_id] = {
            'embedding': embedding,
            'label': label,
            'confidence': None,
            'location': None
        }
        self.next_face_id += 1
        return self.next_face_id - 1

    def log_face_entry(self, name, confidence, embedding):
        if confidence is None:
            return

        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="@Rida12348765",
            database="secure_nest_db"
        )
        cursor = conn.cursor()

        try:
            if name == "Unknown":
                cursor.execute("SELECT face_encoding FROM unauthorized_entries")
                existing_encodings = cursor.fetchall()

                for (stored_encoding_blob,) in existing_encodings:
                    if stored_encoding_blob is None:
                        continue 
                    stored_encoding = np.frombuffer(stored_encoding_blob, dtype=np.float32)
                    similarity = cosine_similarity([embedding], [stored_encoding])[0, 0]
                    if similarity > self.similarity_threshold:
                        print(f"Similar unrecognized face found, skipping entry.")
                        return

                encoding_blob = embedding.astype(np.float32).tobytes()
                cursor.execute("INSERT INTO unauthorized_entries (entry_type, identifier, confidence, face_encoding) VALUES (%s, %s, %s, %s)", 
                               ("face", name, confidence, encoding_blob))
                conn.commit()
                print("Successfully logged entry for unrecognized person.")
                send_email_alert("face (Unknown)", name, confidence)

            else:
                cursor.execute("SELECT * FROM authorized_entries WHERE name = %s", (name,))
                if cursor.fetchone() is None:
                    cursor.execute("INSERT INTO authorized_entries (name, confidence) VALUES (%s, %s)", (name, confidence))
                    conn.commit()
                    print(f"Successfully logged entry for {name}")
                    send_email_alert("face (Authorized)", name, confidence)

        except mysql.connector.Error as err:
            print(f"Database error: {err}")
        finally:
            cursor.close()
            conn.close()

# LicensePlateSystem Class
class LicensePlateSystem:
    def __init__(self, authorized_plates_path, model_path):
        self.plate_detector = YOLO(model_path)
        self.reader = easyocr.Reader(['en'])
        self.authorized_plates = set(pd.read_csv(authorized_plates_path)['license_number'].str.upper().str.replace(' ', ''))
        self.detected_plates = set()

    def detect_and_verify_plates(self, frame):
        results = self.plate_detector.predict(frame)
        unauthorized_plates = []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_crop = frame[y1:y2, x1:x2]
                plate_text, confidence = self.read_license_plate(plate_crop)
                
                if plate_text and confidence is not None:
                    if plate_text not in self.authorized_plates and plate_text not in self.detected_plates:
                        self.detected_plates.add(plate_text)
                        unauthorized_plates.append({
                            'text': plate_text,
                            'confidence': confidence,
                            'bbox': (x1, y1, x2, y2)
                        })
                        self.log_plate_entry(plate_text, confidence)
        return unauthorized_plates

    def read_license_plate(self, plate_crop):
        if plate_crop is None or plate_crop.size == 0:
            return None, None
        try:
            detections = self.reader.readtext(plate_crop)
            if detections:
                text = detections[0][-2].upper().replace(' ', '')
                confidence = float(detections[0][-1])
                if len(text) == 10 and text.isalnum():
                    return text, confidence
            return None, None
        except Exception as e:
            print(f"Error reading license plate: {str(e)}")
            return None, None

    def log_plate_entry(self, plate_text, confidence):
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="@Rida12348765",
            database="secure_nest_db"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM unauthorized_entries WHERE identifier = %s", (plate_text,))
        if cursor.fetchone() is None:
            cursor.execute(
                "INSERT INTO unauthorized_entries (entry_type, identifier, confidence) VALUES (%s, %s, %s)",
                ("license_plate", plate_text, confidence)
            )
            send_email_alert("license_plate (Unauthorized)", plate_text, confidence)
        conn.commit()
        cursor.close()
        conn.close()

# Main Function
def main():
    config = {
        'svm_model_path': r'C:\Users\Lenovo\Desktop\Facerecognition\best_face_classifier.pkl',
        'embeddings_path': r'C:\Users\Lenovo\Desktop\Facerecognition\embeddings.npz',
        'authorized_plates_path': r"C:\Users\Lenovo\Desktop\SecureNest\authorized_plates.csv",
        'license_plate_model_path': r"C:\Users\Lenovo\Desktop\SecureNest\best.pt",
        'video_path': r"C:\Users\Lenovo\Desktop\SecureNest\demo1.mp4",
        'output_path': r'C:\Users\Lenovo\Desktop\Facerecognition\test4sql_annotated_video.mp4'
    }
    
    face_recognizer = FaceRecognizer(config['svm_model_path'], config['embeddings_path'])
    license_plate_system = LicensePlateSystem(config['authorized_plates_path'], config['license_plate_model_path'])
    object_detector = YOLO('yolov8n.pt')
    detected_objects = set()
    
    video_source = cv2.VideoCapture(config['video_path'])
    frame_width = int(video_source.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_source.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(config['output_path'], fourcc, fps, (frame_width, frame_height))

    print("Starting video processing...")
    while True:
        ret, frame = video_source.read()
        if not ret:
            break

        # Recognize faces and draw boxes
        recognized_faces = face_recognizer.recognize_faces(frame)
        for face in recognized_faces:
            name = face['name']
            top, right, bottom, left = face['location']
            confidence = face['confidence']
            
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, f"{name} ({confidence:.2f})", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # License plate detection and annotation
        unauthorized_plates = license_plate_system.detect_and_verify_plates(frame)
        for plate in unauthorized_plates:
            x1, y1, x2, y2 = plate['bbox']
            text = plate['text']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"Unauthorized: {text}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # YOLO object detection and annotation
        results = object_detector.predict(frame)
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = result.names[cls]
                if label not in detected_objects and conf > 0.5:
                    detected_objects.add(label)
                    print(f"New object detected: {label}")
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the frame to the output video
        out.write(frame)

    print("Video processing completed")
    video_source.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    init_db()
    main()
