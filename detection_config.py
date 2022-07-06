import os
import cv2 as cv
import numpy as np


class Config:
    def __init__(self, img_path='IMG_0550.jpg', confidence=0.5, threshold=0.4, img_height=400, img_weight=300):
        self.now_cwd = os.getcwd()
        self.yolo_dir = os.path.join(self.now_cwd, 'yolov3')
        self.config = {
            'weightsPath': os.path.join(self.yolo_dir, 'yolov3.weights'),  # 权重文件
            'configPath': os.path.join(self.yolo_dir, 'yolov3.cfg'),
            'labelsPath': os.path.join(self.yolo_dir, 'coco.names'),
            'imgPath': os.path.join(self.yolo_dir, img_path),
            'confidence': confidence,
            'threshold': threshold,
            'img_height': img_height,
            'img_weight': img_weight,
            'clf_model_path': os.path.join(self.now_cwd, 'pretrain_model/sheepModel8.pt')
        }

    def reload_yolo_net(self, weights_path='yolov3.weights', config_path='yolov3.cfg', labels_path='coco.names'):
        self.config['weightsPath'] = os.path.join(self.yolo_dir, weights_path)
        self.config['configPath'] = os.path.join(self.yolo_dir, config_path)
        self.config['labelsPath'] = os.path.join(self.yolo_dir, labels_path)


class MyYolo:
    def __init__(self, config: Config):
        self.yolo_net = cv.dnn.readNetFromDarknet(config.config['configPath'], config.config['weightsPath'])
        self.layer_outputs_saved = []

    def my_yolo_init(self, img_to_yolo, config: Config):
        # 加载图片、转为blob格式、送入网络输入层
        # img_to_yolo = cv.resize(img_to_yolo, (int(config.config['img_height'] / 2), int(config.config['img_weight'] / 2)), )
        blob_img = cv.dnn.blobFromImage(img_to_yolo, 1.0 / 255.0, (320, 320), None, True, False)

        # 调用setInput函数将图片送入输入层
        self.yolo_net.setInput(blob_img)
        out_info = self.yolo_net.getUnconnectedOutLayersNames()
        layer_outputs = self.yolo_net.forward(out_info)
        self.layer_outputs_saved = layer_outputs

        return layer_outputs


# layerOutputs的第1维的元素内容: [center_x, center_y, width, height, objectness, N-class score data]
class YoloBoxes:
    def __init__(self, config: Config, layers: list, height, weight):
        self.boxes = []
        self.confidences = []
        self.classIDs = []
        self.idxs = np.ndarray([])
        self.layers = layers
        self.config = config.config
        self.H = height
        self.W = weight
        self.labels = self.get_labels()
        self.COLOR = [255, 99, 71]  # orange

    def get_labels(self) -> list:
        with open(self.config['labelsPath'], 'rt') as f:
            labels = f.read().rstrip('\n').split('\n')
        return labels

    def boxes_high_confidence(self):
        for out in self.layers:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.config['confidence']:
                    box = detection[0:4] * np.array([self.W, self.H, self.W, self.H])
                    (centerX, centerY, width, height) = box.astype('int')
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    self.boxes.append([x, y, int(width), int(height)])
                    self.confidences.append(float(confidence))
                    self.classIDs.append(class_id)

        self.idxs = cv.dnn.NMSBoxes(self.boxes, self.confidences, self.config['confidence'], self.config['threshold'])

    def show_boxes(self, img: np.ndarray, classification: int) -> (np.ndarray, list):
        if len(self.idxs) > 0:
            for i in self.idxs.flatten():
                (x, y) = (self.boxes[i][0], self.boxes[i][1])
                (w, h) = (self.boxes[i][2], self.boxes[i][3])

                if self.labels[self.classIDs[i]] == 'dog' or self.labels[self.classIDs[i]] == 'sheep':
                    cv.rectangle(img, (x, y), (x + w, y + h), self.COLOR, 2)
                    text = 'sheep_classification{}: {:.4f}'.format(classification, self.confidences[i])
                    cv.putText(img, text, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR, 2)
                    return img, [x, y, w, h]
        return img, [0, 0, 0, 0]

    def show_boxes_no_classification(self, img: np.ndarray) -> (np.ndarray, list):
        if len(self.idxs) > 0:
            for i in self.idxs.flatten():
                (x, y) = (self.boxes[i][0], self.boxes[i][1])
                (w, h) = (self.boxes[i][2], self.boxes[i][3])

                cv.rectangle(img, (x, y), (x + w, y + h), self.COLOR, 2)
                return img, [x, y, w, h]
                # if self.labels[self.classIDs[i]] == 'dog' or self.labels[self.classIDs[i]] == 'sheep':
                #     cv.rectangle(img, (x, y), (x + w, y + h), self.COLOR, 2)
                #     return img, [x, y, w, h]
        return img, [0, 0, 0, 0]


class Camera:
    def __init__(self, identification=0):
        self.camera_number = identification
        self.cap = cv.VideoCapture(self.camera_number, cv.CAP_DSHOW)

    def get_image(self, weight=640, height=480, image_path=None):
        if self.cap.isOpened():
            # 设置获取的格式 常见格式2560*1440 1920*1080 1280*720
            ret = self.cap.set(3, weight)
            ret = self.cap.set(4, height)
            ret, frame = self.cap.read()
            self.cap.release()
            return 'true', frame
        else:
            print('Haven\'t got image from camera{}'.format(self.camera_number))
            if image_path is not None:
                frame = cv.imread(image_path)
                frame = cv.resize(frame, (weight, height))
                return 'false', frame
            else:
                print('Error: false imgPath in Config!')
                exit()
