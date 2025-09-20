from ultralytics import YOLO
import numpy as np
#model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)

# Load a model
model = YOLO("./best.pt")  # load a fine-tuned model

# Inference using the model (img/video/stream)
""" prediction_results = model("renders/rgb_000.png")
for idx,result in enumerate(prediction_results):
    masks = result.masks.xy  # mask in polygon format
    xyn = result.masks.xyn  # normalized
    masks = result.masks.data 
    break """
prediction_results = model.predict("renders/rgb_006.png",save=True)
