import cv2
import os
import numpy as np
import os.path as osp
from skimage import io
import random
from sklearn import metrics
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import joblib
 # define hog_descriptor function
def hog_descriptor(image):
    # normalize image
    if (image.max()-image.min()) != 0:
        image = (image - image.min()) / (image.max() - image.min())
        image *= 255
        image = image.astype(np.uint8)
    # compute hog descriptor
    hog = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
    hog_feature = hog.compute(image)
    return hog_feature
 # get list of positive and negative images
poslist = os.listdir('E:/dataset/INRIA Person Dataset/person_1/INRIAPerson/Train/pos')
neglist = os.listdir('E:/dataset/INRIA Person Dataset/person_1/INRIAPerson/Train/neg')
testlist = os.listdir('E:/dataset/INRIA Person Dataset/person_1/INRIAPerson/Test/pos')
testnlist = os.listdir('E:/dataset/INRIA Person Dataset/person_1/INRIAPerson/Test/neg')
 # create lists for hog features and labels
hog_list = []
label_list = []
 # print number of positive and negative images
print("正样本图像有"+str(len(poslist)))
print("负样本原始图像有"+str(len(neglist))+"，每个原始图像提供十个负样本")
 # loop through positive images
for i in range(len(poslist)):
    # read image
    posimg = io.imread(osp.join('E:/dataset/INRIA Person Dataset/person_1/INRIAPerson/Train/pos',poslist[i]))
    # convert to BGR
    posimg = cv2.cvtColor(posimg,cv2.COLOR_RGBA2BGR)
    # resize image
    posimg = cv2.resize(posimg, (64, 128), interpolation=cv2.INTER_NEAREST)
    # compute hog feature
    pos_hog = hog_descriptor(posimg)
    # append hog feature and label to lists
    hog_list.append(pos_hog)
    label_list.append(1)
 # loop through negative images
for i in range(len(neglist)):
    # read image
    negimg = io.imread(osp.join('E:/dataset/INRIA Person Dataset/person_1/INRIAPerson/Train/neg',neglist[i]))
    # convert to BGR
    negimg = cv2.cvtColor(negimg, cv2.COLOR_RGBA2BGR)
    # loop through 10 random crops of the negative image
    for j in range(10):
        # generate random coordinates
        y = int(random.random() * (negimg.shape[0] - 128))
        x = int(random.random() * (negimg.shape[1] - 64))
        # crop image
        negimgs = negimg[y:y + 128, x:x + 64]
        # resize image
        negimgs = cv2.resize(negimgs, (64, 128), interpolation=cv2.INTER_NEAREST)
        # compute hog feature
        neg_hog = hog_descriptor(negimgs)
        # append hog feature and label to lists
        hog_list.append(neg_hog)
        label_list.append(0)
 # print type of hog features
print(type(hog_list[10]))
print(type(hog_list[-10]))
 # convert hog features and labels to float32 and int32 respectively
hog_list = np.float32(hog_list)
label_list = np.int32(label_list).reshape(len(label_list),1)
 # train svm model
clf = SVC(C=1.0, gamma='auto', kernel='rbf', probability=True)
clf.fit(hog_list.squeeze(), label_list.squeeze())
# save model
joblib.dump(clf, "E:/hog_svm_test1/result_hog/trained_svm1.m")
 # create lists for test hog features and labels
test_hog = []
test_label = []
 # loop through positive test images
for i in range(len(testlist)):
    # read image
    testimg = io.imread(osp.join('E:/dataset/INRIA Person Dataset/person_1/INRIAPerson/Test/pos', testlist[i]))
    # convert to BGR
    testimg = cv2.cvtColor(testimg, cv2.COLOR_RGBA2BGR)
    # resize image
    testimg = cv2.resize(testimg, (64, 128), interpolation=cv2.INTER_NEAREST)
    # compute hog feature
    testhog = hog_descriptor(testimg)
    # append hog feature and label to lists
    test_hog.append(testhog)
    test_label.append(1)
 # loop through negative test images
for i in range(len(testnlist)):
    # read image
    testnegimg = io.imread(osp.join('E:/dataset/INRIA Person Dataset/person_1/INRIAPerson/Test/neg',testnlist[i]))
    # convert to BGR
    testnegimg = cv2.cvtColor(testnegimg, cv2.COLOR_RGBA2BGR)
    # loop through 10 random crops of the negative image
    for j in range(10):
        # generate random coordinates
        y = int(random.random() * (testnegimg.shape[0] - 128))
        x = int(random.random() * (testnegimg.shape[1] - 64))
        # crop image
        testnegimgs = testnegimg[y:y + 128, x:x + 64]
        # resize image
        testnegimgs = cv2.resize(testnegimgs, (64, 128), interpolation=cv2.INTER_NEAREST)
        # compute hog feature
        testneg_hog = hog_descriptor(testnegimgs)
        # append hog feature and label to lists
        test_hog.append(testneg_hog)
        test_label.append(0)
test_hog = np.float32(test_hog)
test_label = np.int32(test_label).reshape(len(test_label),1)
#可以导入训练后的SVM
clf = joblib.load("E:/hog_svm_test1/result_hog/trained_svm1.m")


#对训练集进行预测并绘制PR、ROC曲线计算AUC值
prob = clf.predict_proba(test_hog.squeeze())[:, 1]

precision, recall, thresholds_1 = metrics.precision_recall_curve(test_label.squeeze(), prob)

plt.figure(figsize=(20, 20), dpi=100)
plt.plot(precision, recall, c='red')
plt.scatter(precision, recall, c='blue')
plt.xlabel("precision", fontdict={'size': 16})
plt.ylabel("recall", fontdict={'size': 16})
plt.title("PR_curve", fontdict={'size': 20})
plt.savefig('E:/hog_svm_test1/PR_try2.png',dpi=300)
Ap=metrics.average_precision_score(test_label.squeeze(), prob)

fpr, tpr, thresholds_2 = metrics.roc_curve(test_label.squeeze(), prob, pos_label=1)

plt.figure(figsize=(20, 20), dpi=100)
plt.plot(fpr, tpr, c='red')
plt.scatter(fpr, tpr, c='blue')
plt.xlabel("FPR", fontdict={'size': 16})
plt.ylabel("TPR", fontdict={'size': 16})
plt.title("ROC_curve", fontdict={'size': 20})
plt.savefig('E:/hog_svm_test1/ROC_try2.png', dpi=300)

AUC=metrics.roc_auc_score(test_label.squeeze(), prob)
print(AUC)
print(Ap)
