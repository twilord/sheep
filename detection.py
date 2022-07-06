import cv2 as cv
from detection_config import Config, MyYolo, Camera, YoloBoxes
from detection_utils import ImageUtils, ClfSheep
from main import Net

if __name__ == '__main__':
    # 加载yolo的配置,加载yolo网络，预训练模型
    config = Config(img_weight=1280, img_height=1280)
    my_yolo = MyYolo(config)

    # 加载摄像头, capture图片，处理异常, img是opencv的格式
    camera_hik = Camera(identification=0)
    (gotten, img) = camera_hik.get_image(weight=config.config['img_weight'], height=config.config['img_height'],
                                         image_path=config.config['imgPath'])

    # pIL 图片预处理， img是pil的格式
    img_l = ImageUtils.img_cv_to_pill(img, True)

    # 分类模型加载
    clf_sheep = ClfSheep(config.config['clf_model_path'])
    ans = clf_sheep.sheep_clf(img_l, config)

    # 拿到图片尺寸
    H = config.config['img_height']
    W = config.config['img_weight']
    (H, W) = img.shape[:2]

    # 找到符合box，绘制
    yolo_output_layers = my_yolo.my_yolo_init(img, config)
    my_yolo_boxes = YoloBoxes(config=config, layers=yolo_output_layers, height=H, weight=W)
    my_yolo_boxes.boxes_high_confidence()
    (img, boxes) = my_yolo_boxes.show_boxes(img=img, classification=ans)
    cv.imshow('detected image', img)
    cv.waitKey(0)
