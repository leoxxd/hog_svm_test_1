import cv2 as cv

# 主程序入口
if __name__ == '__main__':
    # 读取图像
    src = cv.imread("E:/dataset/INRIA Person Dataset/person_1/INRIAPerson/Train/pos/crop_000010.png")
    cv.imshow("input", src)
    # HOG + SVM
    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
    # Detect people in the image
    (rects, weights) = hog.detectMultiScale(src,
                                            winStride=(4, 4),
                                            padding=(8, 8),
                                            scale=1.25,
                                            useMeanshiftGrouping=False)
    # 矩形框
    for (x, y, w, h) in rects:
        cv.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 显示
    cv.imshow("hog-detector", src)
    cv.waitKey(0)
    cv.destroyAllWindows()