from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from skimage import feature, exposure
import joblib
import cv2
import os
import os.path as osp
import numpy as np
def preprocessing(datasetpath):
    
    dataset = []
    labels = []
    
    categories = [os.join(datasetpath, i) for i in os.listdir(datasetpath)]
    
    for i in categories:
        dataset.extend([os.join(i, f) for f in os.listdir(i)])
        
    labels = [i.split("/")[1] for i in dataset]
    return (dataset, labels)
class HOG:
    def __init__(self, orientations = 9, pixelsPerCell = (8, 8),
        cellsPerBlock = (3, 3), transform = False):
        self.orienations = orientations
        self.pixelsPerCell = pixelsPerCell
        self.cellsPerBlock = cellsPerBlock
        self.transform = transform
        
    def describe(self, image):
        hist = feature.hog(image, orientations = self.orienations,
                           pixels_per_cell = self.pixelsPerCell,
                           cells_per_block = self.cellsPerBlock,
                           transform_sqrt = self.transform)
        return hist



(dataset, labels) = preprocessing("E:/dataset/111")
L = LabelEncoder()  # 将n个类别编码为0~n-1之间的整数(包括0和n-1)
labels = L.fit_transform(labels)  # 对labels编码
hog = HOG(transform = True)  # 实例化一个HOG类
data = []
for i in dataset:	# 提取HOG特征，存入data
    image = cv2.imread(i)
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (64, 128), interpolation = cv2.INTER_AREA)
    
    hist = hog.describe(resized)
    data.append(hist)
    
(trainData, testData, trainLabels, testLabels) = train_test_split(
    np.array(data), np.array(labels), test_size=0.15, random_state=42)	# 划分数据集

print("### Training model...")
model = LinearSVC()   # 利用sklearn中的svm进行训练
model.fit(trainData, trainLabels)
print("### Evaluating model...")
predictions = model.predict(testData)
print(classification_report(testLabels, predictions, target_names=L.classes_))
joblib.dump(model, "E:/hog_svm_test1/result_hog/trained_svm3.m")
def sliding_window(image, window = (64, 128), step = 4):
    for y in range(0, image.shape[0] - window[1], step):
        for x in range(0, image.shape[1] - window[0], step):
            yield (x, y, image[y:y + window[1], x:x + window[0]]) 

def pyramid(image, top = (224, 224), ratio = 1.5):
    yield image
    while True:
        (w, h) = (int(image.shape[1] / ratio), int(image.shape[0] / ratio))
        image = cv2.resize(image, (w, h), interpolation = cv2.INTER_AREA)
        if w < top[1] or h < top[0]:
            break
        
        yield image
def resize(image, height = None, width = None):
    h, w = image.shape[:2]
    dim = None
    
    if width is None and height is None:
        return image
    
    if width is None:
        dim = (int(w * (height / h)), height)
    else:
        dim = (width, int(h * (width / w)))
        
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized
def coordinate_transformation(height, width, h, w, x, y, roi):
    if h is 0 or w is 0:
        print("divisor can not be zero !!")
    
    img_h = int(height/h * roi[1])
    img_w = int(width/w * roi[0])

    img_y = int(height/h * y)
    img_x = int(width/w * x)

    return (img_x, img_y, img_w, img_h) 
import joblib 

img_path = "man.jpg"

""" 图像金字塔与滑动窗口参数 """
ratio = 1.5
i_roi = (64, 128)
step = 20
top = (128, 128)

model = joblib.load("E:/hog_svm_test1/result_hog/trained_svm3.m")   # 加载模型对象
hog = HOG(transform = True)	   # 实例化HOG对象，并将 transform 参数设为 True

image = cv2.imread(img_path)
resized = resize(image, height = 500)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
height, width = gray.shape[:2]


roi_loc = []	# 空列表，为了存储人的特征信息
for image in pyramid(gray, top = (128, 128), ratio = ratio):	# 每层滑动窗口进行resize，然后使用HOG将提取到的特征传给hist
    h, w = image.shape[:2]
    
    for (x, y, roi) in sliding_window(image, window = i_roi, step = step):
        roi = cv2.resize(roi, (64, 128), interpolation = cv2.INTER_AREA)   
        hist = hog.describe(roi)
        
        if model.predict([hist])[0]:	# 如果预测结果为真，也就是1，将进行下一步坐标变换
            img_x, img_y, img_w, img_h = coordinate_transformation(height, width, h, w, x, y, i_roi) 
            roi_loc.append([img_x, img_y, img_w, img_h])	# 将顶点坐标、宽、高存入上述rol_loc中，等待标定

def NMS(boxes, threshold):
    if len(boxes) == 0:		# 边界判断，如果没有检测到任何目标，返回空列表，即不做nms
        return []
    
    boxes = np.array(boxes).astype("float")	# 将numpy中的每个元素转换成float类型

    x1 = boxes[:,0]  # 左上角顶点的横坐标
    y1 = boxes[:,1]	 # 左上角顶点的纵坐标
    w1 = boxes[:,2]  # 矩形框的宽
    h1 = boxes[:,3]  # 矩形框的高
    x2 = x1 + w1  # 右下角顶点横坐标的集合
    y2 = y1 + h1  # 纵坐标的集合
    
    area = (w1 + 1) * (h1 + 1)  # 计算每个矩形框的面积，这里分别加1是为了让IOU匹配不出现0的情况
    temp = []
    
    idxs = np.argsort(h1)	# 将 h1 中的元素从小到大排序并返回每个元素在 h1 中的下标
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        temp.append(i)   
        
        x1_m = np.maximum(x1[i], x1[idxs[:last]])	# 将其他矩形框的左上角横坐标两两比较
        y1_m = np.maximum(y1[i], y1[idxs[:last]])   # 其他矩形框的左上角纵坐标两两比较
        """两个矩形框重叠的部分是矩形，这一步的目的是为了找到这个重叠矩形的左上角顶点"""
        
        x2_m = np.minimum(x2[i], x2[idxs[:last]])	
        y2_m = np.minimum(y2[i], y2[idxs[:last]])
        """目的是为了找出这个重叠矩形的右下角顶点"""

        w = np.maximum(0, x2_m - x1_m + 1)	# 计算矩形的宽
        h = np.maximum(0, y2_m - y1_m + 1)	# 计算矩形的高
        """剔除掉没有相交的矩形，因为两个矩形框相交，则 x2_m - x1_m + 1 和 y2_m - y1_m + 1 大于零，如果两个矩形框不相交则这两个值小于零"""
        
        over = (w * h) / area[idxs[:last]]	# 计算重叠矩形面积和 area 中的面积的比值
        
        idxs = np.delete(idxs, np.concatenate(([last],  	
            np.where(over > threshold)[0])))  	# 剔除重叠的矩形框

    return boxes[temp].astype("int")
for (x, y, w, h) in NMS(roi_loc, threshold=0.3):
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
from matplotlib import pyplot as plt
# %matplotlib inline

plt.figure(figsize = (10,10))
resized = resized[:,:,::-1]
plt.imshow(resized)
