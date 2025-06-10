import face_recognition
import os
import csv

def save_encodings_to_csv(dataset_path, csv_file):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)

        # Loop through each person in the dataset
        for person_name in os.listdir(dataset_path):
            person_folder = os.path.join(dataset_path, person_name)
            if os.path.isdir(person_folder):
                for image_name in os.listdir(person_folder):
                    image_path = os.path.join(person_folder, image_name)
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        encoding = encodings[0]
                        row = [person_name] + encoding.tolist()  # Store name and encodings in one row
                        writer.writerow(row)  # Write row to the CSV

# Make sure the dataset path points to your folder with images
dataset_path = 'C:/Users/asus/OneDrive/Desktop/Mini Project/Test Subjects/Dataset C'
csv_file = 'face_encodings.csv'
save_encodings_to_csv(dataset_path, csv_file)

print("Face encodings saved to face_encodings.csv.")
