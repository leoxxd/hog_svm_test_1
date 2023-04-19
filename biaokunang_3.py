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

model = joblib.load("model")   # 加载模型对象
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

