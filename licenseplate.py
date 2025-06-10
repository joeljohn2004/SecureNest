!pip install ultralytics
!pip install roboflow
!pip install easyocr
!pip install opencv-python-headless  # Use headless for environments like Colab where display is not needed
!pip install pandas
!pip install numpy

import cv2
from ultralytics import YOLO
import easyocr
import pandas as pd
import numpy as np
import time
import os

class LicensePlateSystem:
    def __init__(self, authorized_plates_path):
        """
        Initialize the license plate detection system
        
        Args:
            authorized_plates_path: Path to CSV file containing authorized license plates
        """
        # Initialize models
        self.plate_detector = YOLO('/content/drive/MyDrive/license_plate_model/best.pt')  # Adjust if using different model
        self.reader = easyocr.Reader(['en'])  # Change to other languages if needed
        
        # Load authorized plates
        self.authorized_plates = pd.read_csv(authorized_plates_path)['license_number'].tolist()
        
        # Results storage
        self.results = []

    def read_license_plate(self, plate_crop):
        """Read text from license plate image"""
        detections = self.reader.readtext(plate_crop)
        if not detections:
            return None, None
            
        text = detections[0][-2].upper().replace(' ', '')
        score = detections[0][-1]
        
        # Basic license plate format validation (customize based on your needs)
        if len(text) >= 5 and text.isalnum():
            return text, score
        return None, None

    def process_video(self, video_path, output_csv, output_video):
        """Process video and detect license plates"""
        cap = cv2.VideoCapture('/content/demo (1).mp4')
        frame_number = 0
        
        # Check if video file opened successfully
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return
        
        # Get the video frame width and height for output video writer
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create video writer to save output video with detections
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_number += 1
            
            # Detect license plates
            detections = self.plate_detector(frame)[0]
            
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                
                if score > 0.5:  # Confidence threshold
                    # Crop license plate
                    plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    
                    # Read license plate text
                    plate_text, text_score = self.read_license_plate(plate_crop)
                    
                    if plate_text and text_score > 0.5:  # Confidence threshold for OCR
                        # Check if plate is authorized
                        is_authorized = plate_text in self.authorized_plates
                        
                        # Store results
                        self.results.append({
                            'frame_number': frame_number,
                            'license_number': plate_text,
                            'confidence': text_score,
                            'bbox': f"[{x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}]",
                            'is_authorized': is_authorized
                        })
                        
                        # Draw bounding box and plate text on frame
                        color = (0, 255, 0) if is_authorized else (0, 0, 255)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(frame, plate_text, (int(x1), int(y1)-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Save the processed frame to the output video
            out.write(frame)

        cap.release()
        out.release()  # Release the video writer
        
        # Save results to CSV
        df = pd.DataFrame(self.results)
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

        total_time = time.time() - start_time
        print(f"Processed {frame_number} frames in {total_time:.2f} seconds.")


# Example usage
def main():
    # Create authorized plates CSV (example)
    authorized_plates = pd.DataFrame({
        'license_number': ['N894JV', 'XYZ789', 'H644LX']
    })
    authorized_plates.to_csv('authorized_plates.csv', index=False)
    
    # Initialize and run the system
    system = LicensePlateSystem('authorized_plates.csv')
    system.process_video('/content/sample.mp4', 'results.csv', 'output_with_detections.mp4')

if __name__ == "__main__":
    main()
