!pip install ultralytics
!pip install roboflow
!pip install PyYAML>=5.3.1
!pip install opencv-python>=4.1.1
!pip install numpy>=1.18.5
!pip install torch>=1.7.0 torchvision>=0.8.1

import os
os.environ["WANDB_DISABLED"] = "true"

from roboflow import Roboflow
from ultralytics import YOLO
import os
from google.colab import drive

rf = Roboflow(api_key="qRmYMtUFydtpZ3eGJAV2")
project = rf.workspace("mochoye").project("license-plate-detector-ogxxg")
version = project.version(2)
dataset = version.download("yolov8")

model = YOLO('yolov8n.pt')
model.train(data='/content/License-Plate-Detector-2/data.yaml', epochs=50, imgsz=640)
