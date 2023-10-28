#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 15:20:45 2023

@author: nehal
"""

# Import the necessary packages
import cv2
import imutils
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import streamlit as st

# Load the face detector model
prototxtPath = "/Users/nehal/fip-project-folders/stream-fip/deploy.prototxt"
weightsPath = "/Users/nehal/fip-project-folders/stream-fip/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load the face mask detector model
maskNet = load_model("/Users/nehal/fip-project-folders/stream-fip/mask_detector.model")

# Initialize the video stream
vs = VideoStream(src=0).start()

# Define a Streamlit app
st.title("Face Mask Detection with Streamlit")


# Function to detect and display face mask detection results
def detect_and_display_mask(frame, faceNet, maskNet):
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    return frame


def detect_and_predict_mask(frame, faceNet, maskNet):
    # Grab the dimensions of the frame and then construct a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                (104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # Initialize lists for faces, locations, and predictions
    faces = []
    locs = []
    preds = []

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

# Main Streamlit loop
def tr_function():
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
    
        frame = detect_and_display_mask(frame, faceNet, maskNet)
    
        st.image(frame, use_column_width=True)
    
        if st.button("Exit"):
            break

if st.button('detect the mask'):
    tr_function()

# Release the video stream
vs.stop()



