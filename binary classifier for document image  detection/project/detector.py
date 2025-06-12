import joblib
import cv2 # type: ignore
import numpy as np
import os


model = joblib.load("my_model.pkl")  

def detect_forgery(filepath):
    
    image = cv2.imread(filepath)

    
    resized = cv2.resize(image, (224, 224))  
    flattened = resized.flatten().reshape(1, -1)  

    
    prediction = model.predict(flattened)[0]

    
    label = "forged" if prediction == 1 else "authentic"  

    return {
        "status": label,
        "image_path": filepath  
    }
