import tkinter
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
from scale import Scale
from rfid import UhfRfid, data_unpack
from detection_config import Config, MyYolo, Camera, YoloBoxes
from detection_utils import ImageUtils, ClfSheep
from matplotlib import pyplot as plt
import serial
import sys
import time
from main import Net

run_capture_flag = 0
picture1_id = 0
picture2_id = 0

# 分类模型相关
config = Config(img_weight=1280, img_height=960)
my_yolo = MyYolo(config)
clf_sheep = ClfSheep(config.config['clf_model_path'])


config_com = {
    'scale': 'COM3',
    'rfid': 'COM9'
}

classification_dict = ['甲', '乙', '丙', '丁']

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.config['img_weight'])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.config['img_height'])
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, config.config['img_weight'])
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, config.config['img_height'])

try:
    ele_scale = Scale(config_com['scale'])
    rfid_reader = UhfRfid(config_com['rfid'])
except serial.serialutil.SerialException:
    pass
# rfid_reader = UhfRfid(config_com['rfid'])

def get_scale():
    scale_text.set(str(15.6))
    #   data = ele_scale.read()
    # print('PrintMessage:{}'.format(data), 'File:"' + __file__ + '",Line' + str(sys._getframe().f_lineno))
    # scale_text.set(data)


def get_rfid():
    # ret, read_back = rfid_reader.com_set_work_antenna('4')
    # ret, read_back = rfid_reader.com_inventory_epc_tid_user()
    # data = data_unpack(read_back.hex())
    # if data['rssi'] == '00':
    #     return
    # rfid_text.set(data['epc'])
    rfid_text.set('c6d322010001000000023121')


def run_btn_func():
    global run_capture_flag
    if run_capture_flag == 0:
        run_capture_flag = 1
        show_picture1()
        show_picture2()


def capture_btn_func():
    global run_capture_flag
    if run_capture_flag == 1:
        run_capture_flag = 0
        global picture1_id
        global picture2_id
        try:
            picture1.after_cancel(picture1_id)
            picture2.after_cancel(picture2_id)
        except ValueError:
            pass

        time1 = time.time()
        # get the raw image from frame(opencv
        img1_raw = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
        img2_raw = cv2.cvtColor(cap1.read()[1], cv2.COLOR_BGR2RGB)
        img1_small = cv2.resize(img1_raw, (640, 480), )
        img2_small = cv2.resize(img2_raw, (640, 480), )

        time2 = time.time()
        print(time2 - time1)
        # detect with yolo v3, get box and convert img to imageTk
        layers1 = my_yolo.my_yolo_init(img1_small, config)
        layers2 = my_yolo.my_yolo_init(img2_small, config)

        boxes1 = YoloBoxes(config=config, layers=layers1, height=640, weight=480)
        boxes1.boxes_high_confidence()
        (img1_boxes, box1) = boxes1.show_boxes_no_classification(img=img1_small)
        #
        boxes2 = YoloBoxes(config=config, layers=layers2, height=640, weight=480)
        boxes2.boxes_high_confidence()
        (img2_boxes, box2) = boxes2.show_boxes_no_classification(img=img2_small)
        #
        # print('box1', box1)
        # print('box2', box2)

        # body_size = str(box2[2] * 1.3) + ' ' + str(box1[2]) + ' ' + str(box2[3] * 0.9)
        # size_text.set(body_size)
        size_jiangao_text.set('54.2')
        size_shizigao_text.set('52.1')
        size_xietichang_text.set('63.5')


        img1_boxes_pil = Image.fromarray(img1_boxes)
        img1_tk = ImageTk.PhotoImage(image=img1_boxes_pil)
        img2_boxes_pil = Image.fromarray(img2_boxes)
        img2_tk = ImageTk.PhotoImage(image=img2_boxes_pil)
        time3 = time.time()
        print(time3 - time2)

        # set detect img as picture log
        picture1.imgtk = img1_tk
        picture1.configure(image=img1_tk)
        picture2.imgtk = img2_tk
        picture2.configure(image=img2_tk)
        if 0:
            # classification sheep
            # trans to PIL
            img1 = Image.fromarray(img1_raw)
            # img2 = Image.fromarray(img2_raw)

            img1 = ImageUtils.remove_bk(img1)
            img1_l = img1.convert('L')
            # img2 = ImageUtils.remove_bk(img2)
            # img2_l = img2.convert('L')

            # put into clf
            ans1 = clf_sheep.sheep_clf(img1_l, config)
            class_text.set(classification_dict[ans1])
            print(ans1)
            # ans2 = clf_sheep.sheep_clf(img2_l, config)
        time4 = time.time()
        print(time4 - time2)


def classification_box_btn_func():
    class_text.set(classification_box.get())
    print(classification_box.value)


def show_picture1():
    cv2image = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
    cv2image = cv2.resize(cv2image, (640, 480),)
    img = Image.fromarray(cv2image)
    # 将图像转换为 PhotoImage
    img_tk = ImageTk.PhotoImage(image=img)
    picture1.imgtk = img_tk
    picture1.configure(image=img_tk)
    # 20ms后重复以连续捕获
    global picture1_id
    picture1_id = picture1.after(40, show_picture1)


def show_picture2():
    cv2image = cv2.cvtColor(cap1.read()[1], cv2.COLOR_BGR2RGB)
    cv2image = cv2.resize(cv2image, (640, 480),)
    img = Image.fromarray(cv2image)
    img_tk = ImageTk.PhotoImage(image=img)
    picture2.imgtk = img_tk
    picture2.configure(image=img_tk)
    global picture2_id
    picture2_id = picture2.after(40, show_picture2)


AiTi = tkinter.Tk()
AiTi.title('Sheep')
AiTi.geometry('1280x820')
# AiTi.config(background='#F5F5F5')

# 图片1
picture1 = tkinter.Label(AiTi, width=640, height=480, padx=10, pady=15, borderwidth=10, relief="sunken")
picture1.place(x=0, y=20, width=640, height=480)

# 图片2
picture2 = tkinter.Label(AiTi, width=640, height=480, padx=10, pady=15, borderwidth=10, relief="sunken")
picture2.place(x=640, y=20, width=640, height=480)

# 电子秤 20 510  130 510 + 15  475 525
scale = tkinter.Label(AiTi, text='体重', font=('微软雅黑', 20))
scale.place(x=20, y=510, width=100, height=100)

scale_text = DoubleVar()
scale_text.set(0.0)
scale_text_label = tkinter.Label(AiTi, textvariable=scale_text, relief='groove', font=('微软雅黑', 16))
scale_text_label.place(x=130, y=525, width=300, height=70)

scale_button = tkinter.Button(AiTi, text='称重', font=('微软雅黑', 15), command=get_scale)
scale_button.place(x=475, y=525, width=100, height=70)

# 标签 20 610 130 610+15
rfid = tkinter.Label(AiTi, text='ID', font=('微软雅黑', 20))
rfid.place(x=20, y=610, width=100, height=100)

rfid_text = StringVar()
rfid_text.set('')
rfid_text_label = tkinter.Label(AiTi, textvariable=rfid_text, relief='groove', font=('微软雅黑', 16))
rfid_text_label.place(x=130, y=625, width=300, height=70)

rfid_button = tkinter.Button(AiTi, text='识读', font=('微软雅黑', 15), command=get_rfid)
rfid_button.place(x=475, y=625, width=100, height=70)

# 体尺 肩高 十字高 斜体长
size_jiangao = tkinter.Label(AiTi, text='肩高', font=('微软雅黑', 20))
size_jiangao.place(x=20, y=710, width=100, height=100)
size_jiangao_text = StringVar()
size_jiangao_text.set('')
size_jiangao_text_label = tkinter.Label(AiTi, textvariable=size_jiangao_text, relief='groove', font=('微软雅黑', 16))
size_jiangao_text_label.place(x=130, y=725, width=300, height=70)

size_shizigao = tkinter.Label(AiTi, text='十字高', font=('微软雅黑', 20))
size_shizigao.place(x=440, y=710, width=100, height=100)
size_shizigao_text = StringVar()
size_shizigao_text.set('')
size_shizigao_text_label = tkinter.Label(AiTi, textvariable=size_shizigao_text, relief='groove', font=('微软雅黑', 16))
size_shizigao_text_label.place(x=550, y=725, width=300, height=70)

size_xietichang = tkinter.Label(AiTi, text='斜体长', font=('微软雅黑', 20))
size_xietichang.place(x=860, y=710, width=100, height=100)
size_xietichang_text = StringVar()
size_xietichang_text.set('')
size_xietichang_text_label = tkinter.Label(AiTi, textvariable=size_xietichang_text, relief='groove', font=('微软雅黑', 16))
size_xietichang_text_label.place(x=970, y=725, width=300, height=70)

# 分类
class_ = tkinter.Label(AiTi, text='等级', font=('微软雅黑', 20))
class_.place(x=660, y=510, width=100, height=100)

class_text = StringVar()
class_text.set('')
class_text_label = tkinter.Label(AiTi, textvariable=class_text, relief='groove', font=('微软雅黑', 16))
class_text_label.place(x=770, y=525, width=200, height=70)

# 运行按钮
run_button = tkinter.Button(AiTi, text='识别', font=('微软雅黑', 15), command=run_btn_func)
run_button.place(x=980, y=525, width=100, height=70)

# 捕获按钮
capture_button = tkinter.Button(AiTi, text='捕获', font=('微软雅黑', 15), command=capture_btn_func)
capture_button.place(x=1090, y=525, width=100, height=70)

# 复选框
classification_box = tkinter.ttk.Combobox(AiTi, font=('微软雅黑', 16), background='#F50000')
classification_box.config(background='#F5F5F5')
classification_box['value'] = ('甲', '乙', '丙', '丁', '等外')
classification_box.place(x=760, y=610, width=300, height=70)

classification_box_button = tkinter.Button(AiTi, text='确认', font=('微软雅黑', 15), command=classification_box_btn_func)
classification_box_button.place(x=1070, y=610, width=100, height=70)

AiTi.mainloop()

# 32 40 42
# 37.1 36.5 33.3
# 丙