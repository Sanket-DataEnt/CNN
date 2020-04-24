#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 23:36:56 2020

@author: sanket
"""


import cv2
import numpy as np

# read pre-trained model and config file and creating the network.
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")


# Read the text file containing class names in human readable form and extract the class names to a list.
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
print (classes)

# To get the output layers name
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# generate different colors for different classes to draw bounding boxes. 
colors = np.random.uniform(0, 255, size=(len(classes), 3))



img = cv2.imread("yolo.jpg")
print(img.shape)
img = cv2.resize(img, None, fx=0.2, fy=0.2) #resizing the image
print(img.shape)

#Reading the dimensions of the input image
height, width, channels = img.shape

# preparing the input image to run through the deep neural network.
# create input blob 
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False) #True converts BGR to RGB format
# set input blob for the network
net.setInput(blob)

# run inference through the network and gather predictions from output layers
outs = net.forward(output_layers)


# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4


# for each detection from each output layer get the confidence, class id, bounding box params and ignore weak detections (confidence < 0.5)
# detection[0:4] represent coordinates ,width and height. detection[5: ] represent the probability of 80 objects.

# loop over each of the layer outputs
for out in outs:
    # loop over each of the detections
    for detection in out:
        # extract the class ID and confidence (i.e., probability) of the current object detection.
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        # filter out weak predictions by ensuring the detected probability is greater than the minimum probability
        if confidence > 0.5:
            # Object detected
            # scale the bounding box coordinates back relative to the size of the image, keeping in mind that YOLO actually
            # returns the center (x, y)-coordinates of the bounding box followed by the boxes' width and height.
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            
            # Rectangle coordinates
            # use the center (x, y)-coordinates to derive the top and left corner of the bounding box
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)


# apply non-maxima suppression to suppress weak, overlapping bounding boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)


# draw bounding box on the detected object with class name
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])+","+ str(format(confidences[i],'.2f'))
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 2, color, 2)


# printing the output

#cv2.imshow('Image',img)
#cv2.waitKey(0)
cv2.imwrite('yolo_predicted.jpg',img)



