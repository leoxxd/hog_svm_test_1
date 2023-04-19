import cv2
import os
import numpy as np
from sklearn.svm import SVC

def extractHOGFeatures(img):
    hog = cv2.HOGDescriptor()
    h = hog.compute(img)
    return h.flatten()

def loadTrainingData():
    pos_dir = 'C:/Users/Leo/Downloads/Compressed/train_64x128_H96/posPatches'
    neg_dir = 'C:/Users/Leo/Downloads/Compressed/train_64x128_H96/negPatches'
    pos_files = os.listdir(pos_dir)
    neg_files = os.listdir(neg_dir)
    X = []
    Y = []
    for f in pos_files:
        img = cv2.imread(os.path.join(pos_dir, f))
        X.append(extractHOGFeatures(img))
        Y.append(1)
    for f in neg_files:
        img = cv2.imread(os.path.join(neg_dir, f))
        X.append(extractHOGFeatures(img))
        Y.append(0)
    return X, Y

def trainSVM(X, Y):
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X, Y)
    return clf

def detectPedestrians(img, clf):
    scale = 1.05
    stride = 8
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(clf.coef_.ravel())
    found, w = hog.detectMultiScale(img, winStride=(8, 8), scale=1.05)
    found_filtered = []
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and np.sum((r - q)**2) <= 25:
                break
        else:
            found_filtered.append(r)
    return found_filtered

def nonMaximumSuppression(boundingBoxes, overlapThreshold):
    if len(boundingBoxes) == 0:
        return []
    if boundingBoxes.dtype.kind == "i":
        boundingBoxes = boundingBoxes.astype("float")
    pick = []
    x1 = boundingBoxes[:,0]
    y1 = boundingBoxes[:,1]
    x2 = boundingBoxes[:,2]
    y2 = boundingBoxes[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]
        for pos in range(0, last):
            j = idxs[pos]
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            overlap = float(w * h) / area[j]
            if overlap > overlapThreshold:
                suppress.append(pos)
        idxs = np.delete(idxs, suppress)
    return boundingBoxes[pick].astype("int")

if __name__ == '__main__':
    X, Y = loadTrainingData()
    clf = trainSVM(X, Y)

    test_dir = 'C:/Users/Leo/Downloads/Compressed/test_64x128_H96'
   
for f in os.listdir(test_dir):
    img = cv2.imread(os.path.join(test_dir, f))
    boxes = detectPedestrians(img, clf)
    boxes = np.array([[x, y, x+w, y+h] for (x, y, w, h) in boxes])
    boxes = nonMaximumSuppression(boxes, 0.3)
    for (x, y, x2, y2) in boxes:
        cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('Pedestrian Detection', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
