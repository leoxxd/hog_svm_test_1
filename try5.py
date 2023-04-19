import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 读取正负样本文件列表
pos_dir = 'C:/Users/Leo/Downloads/Compressed/train_64x128_H96/posPatches'
pos_files = [os.path.join(pos_dir, f) for f in os.listdir(pos_dir)]
neg_dir = 'C:/Users/Leo/Downloads/Compressed/train_64x128_H96/negPatches'
neg_files = [os.path.join(neg_dir, f) for f in os.listdir(neg_dir)]

# 提取正负样本的HOG特征
hog_features = []
labels = []
for f in pos_files:
    img = cv2.imread(f)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_feature = hog(img_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
    hog_features.append(hog_feature)
    labels.append(1)
for f in neg_files:
    img = cv2.imread(f)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_feature = hog(img_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
    hog_features.append(hog_feature)
    labels.append(0)

# 训练SVM模型
clf = LinearSVC()
clf.fit(hog_features, labels)

# 在测试集上进行行人检测并标出框
test_dir = r'C:/Users/Leo/Downloads/Compressed/test_64x128_H96/posPatches'
test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir)]
for f in test_files:
    img = cv2.imread(f)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boxes = []
    for scale in [1.0, 1.5, 2.0]:
        resized = cv2.resize(img_gray, (int(img_gray.shape[1] / scale), int(img_gray.shape[0] / scale)))
        for y in range(0, resized.shape[0] - 128, 8):
            for x in range(0, resized.shape[1] - 64, 8):
                window = resized[y:y+128, x:x+64]
                hog_feature = hog(window, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
                score = clf.decision_function([hog_feature])[0]
                if score > 0.5:
                    box = (int(x*scale), int(y*scale), int(64*scale), int(128*scale))
                    boxes.append(box)

    # 使用NMS算法选出最佳框并绘制
    if len(boxes) > 0:
        boxes = np.array(boxes)
        pick = []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        scores = clf.decision_function(hog_features)
        indices = np.argsort(scores)
        while len(indices) > 0:
            last = len(indices) - 1
            i = indices[last]
            pick.append(i)
            xx1 = np.maximum(x1[i], x1[indices[:last]])
            yy1 = np.maximum(y1[i], y1[indices[:last]])
            xx2 = np.minimum(x2[i], x2[indices[:last]])
            yy2 = np.minimum(y2[i], y2[indices[:last]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            overlap = w * h / (boxes[i, 2] * boxes[i, 3] + boxes[indices[:last], 2] * boxes[indices[:last], 3] - w * h)
            indices = indices[np.where(overlap <= 0.5)[0]]

        for i in pick:
            cv2.rectangle(img, (boxes[i][0], boxes[i][1]), (boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]), (0, 255, 0), 2)

    # 显示并保存结果
    # cv2.imshow('result', img)
    # cv2.waitKey()
    # # cv2.imwrite(f.replace('Test', 'result'), img)

# 计算ROC曲线和AUC
hog_features = []
labels = []
test_dir = r'C:/Users/Leo/Downloads/Compressed/test_64x128_H96/'
test_files = [os.path.join(test_dir, f).replace('\\','/') for f in os.listdir(test_dir)]
for f in test_files:
    if 'posPatches' in f:
        labels.append(1)
    else:
        labels.append(0)
    img = cv2.imread(f,cv2.IMREAD_UNCHANGED)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_feature = hog(img_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
    hog_features.append(hog_feature)
scores = clf.decision_function(hog_features)
fpr, tpr, thresholds = roc_curve(labels, scores)
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('C:/Users/Leo/Downloads/Compressed/test_64x128_H96/ROC.png')
