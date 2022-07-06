import numpy as np
import cv2 as cv
import cv2
import os
import torch
import torchvision.models as models
from main import Net

now_cwd = os.getcwd()
yolo_dir = os.path.join(now_cwd, 'yolov3')

config = {
    'weightsPath':  os.path.join(yolo_dir, 'yolov3.weights'),  # 权重文件
    'configPath': os.path.join(yolo_dir, 'yolov3.cfg'),
    'labelsPath': os.path.join(yolo_dir, 'coco.names'),
    'confidence': 0.5,
    'threshold': 0.4,
    'img_height': 400,
    'img_weight': 300
}

model = torch.load(os.path.join(now_cwd, 'pretrain_model/sheepModel4.pt'))


def get_image():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        # cv2.imshow('origin', frame)
        cap.release()
        return 'true', frame
    else:
        return 'false', 'no_image'


def my_yolo_init(gotten, img):
    net = cv.dnn.readNetFromDarknet(config['configPath'], config['weightsPath'])  # 加载网络、配置权重

    if gotten == 'false':
        return

    # 加载图片、转为blob格式、送入网络输入层
    img = cv.resize(img, (config['img_height'], config['img_weight']))
    blobImg = cv.dnn.blobFromImage(img, 1.0 / 255.0, (416, 416), None, True, False)  # net需要的输入是blob格式的，用blobFromImage这个函数来转格式
    net.setInput(blobImg)  # # 调用setInput函数将图片送入输入层

    outInfo = net.getUnconnectedOutLayersNames()  # 前面的yolov3架构也讲了，yolo在每个scale都有输出，outInfo是每个scale的名字信息，供net.forward使用
    layerOutputs = net.forward(outInfo)  # 得到各个输出层的、各个检测框等信息，是二维结构。
    return layerOutputs


(gotten, img) = get_image()
# 拿到图片尺寸
H = config['img_height']
W = config['img_weight']
# 过滤layerOutputs
# layerOutputs的第1维的元素内容: [center_x, center_y, width, height, objectness, N-class score data]
# 过滤后的结果放入：
boxes = []  # 所有边界框（各层结果放一起）
confidences = []  # 所有置信度
classIDs = []  # 所有分类ID
layerOutputs = my_yolo_init(gotten, img)


# 1）过滤掉置信度低的框框
for out in layerOutputs:  # 各个输出层
    for detection in out:  # 各个框框
        # 拿到置信度
        scores = detection[5:]  # 各个类别的置信度
        classID = np.argmax(scores)  # 最高置信度的id即为分类id
        confidence = scores[classID]  # 拿到置信度

        # 根据置信度筛查
        if confidence > config['confidence']:
            box = detection[0:4] * np.array([W, H, W, H])  # 将边界框放会图片尺寸
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

# 2）应用非最大值抑制(non-maxima suppression，nms)进一步筛掉
idxs = cv.dnn.NMSBoxes(boxes, confidences, config['confidence'], config['threshold'])  # boxes中，保留的box的索引index存入idxs

# 得到labels列表
with open(config['labelsPath'], 'rt') as f:
    labels = f.read().rstrip('\n').split('\n')
# 应用检测结果
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")   # 框框显示颜色，每一类有不同的颜色，每种颜色都是由RGB三个值组成的，所以size为(len(labels), 3)

if len(idxs) > 0:
    for i in idxs.flatten():  # idxs 是二维的，第0维是输出层，所以这里把它展平成1维
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        color = [int(c) for c in COLORS[classIDs[i]]]
        cv.rectangle(img, (x, y), (x+w, y+h), color, 2)  # 线条粗细为2px
        text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
        # text = 'sheep'
        cv.putText(img, text, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)   # cv.FONT_HERSHEY_SIMPLEX字体风格、0.5字体大小、粗细2px

cv.imshow('detected image', img)
cv.waitKey(0)