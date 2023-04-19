# Importing libraries
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
 # Parsing arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="E:/dataset/INRIA Person Dataset/person_1/INRIAPerson/Test/pos")
args = vars(ap.parse_args())
 # Setting up HOG descriptor
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
 # Iterating through images
for imagePath in paths.list_images(args["images"]):
    # Reading image
    image = cv2.imread(imagePath)
    # Resizing image
    image = imutils.resize(image, width=min(400, image.shape[1]))
    # Making a copy of the original image
    orig = image.copy()
    # Detecting people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
        padding=(8, 8), scale=1.05)
    # Drawing bounding boxes around detected people
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # Applying non-maxima suppression
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
     # Drawing bounding boxes around detected people after applying non-maxima suppression
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
     # Getting the filename
    filename = imagePath[imagePath.rfind("/") + 1:]
    # Printing the number of boxes before and after applying non-maxima suppression
    print("[INFO] {}: {} original boxes, {} after suppression".format(
        filename, len(rects), len(pick)))
     # Showing the images
    cv2.imshow("Before NMS", orig)
    cv2.imshow("After NMS", image)
    cv2.waitKey("q")