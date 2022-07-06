from detection_config import Config
from rembg import remove
from rembg.detect import ort_session
import time
import cv2 as cv
import os
import math
from PIL import Image
from torchvision import transforms
import torch
import numpy
from main import Net


# 分割图片
class ClfData:
    def __init__(self, clf_img, cfg: Config, overlap_rate=0.5, size=160):
        self.img = clf_img
        self.height = cfg.config['img_height']
        self.weight = cfg.config['img_weight']
        self.overlap_rate = overlap_rate
        self.list = []
        self.size = size
        self.make_list()

    def make_list(self):
        size_rate = int(self.size * self.overlap_rate)
        for X in range(self.weight)[::size_rate]:
            for Y in range(self.height)[::size_rate]:
                if X + self.size > self.weight or Y + self.size > self.height:
                    continue
                else:
                    self.list.append((X, Y))

    def show_list(self):
        print("data list", self.list)

    def has_item(self):
        if self.list:
            return True
        else:
            return False

    def get_item(self):
        if self.has_item():
            (X, Y) = self.list[0]
            item = self.img.crop((X, Y, X + self.size, Y + self.size))
            self.list.pop(0)
            return item
        else:
            print("No item has in!")
            return


class ImageUtils:
    # 背景去除
    @staticmethod
    def remove_bk(img_to_rembg) -> Image.Image:
        pic = remove(img_to_rembg, session=ort_session("u2netp"))
        pic = pic.convert('RGB')
        return pic

    # 将opencv的图片进行保存, img_name_save
    @staticmethod
    def img_save_cv(img_path_save, img_to_save, img_name_save=None):
        if img_name_save is None:
            img_name_save = 'picture/OG{}.jpg'.format(math.ceil(time.time()))
        cv.imwrite(os.path.join(img_path_save, img_name_save), img_to_save)

    # 将PIL的图片进行保存
    @staticmethod
    def img_save_pil(img_path_save, img_to_save, img_name_save=None):
        if img_name_save is None:
            img_name_save = 'picture/PIL{}.jpg'.format(math.ceil(time.time()))
        img_to_save.save(os.path.join(img_path_save, img_name_save), img_to_save)

    @staticmethod
    def img_cv_to_pill(img, remove_bk: bool = False) -> Image.Image:
        # 图片格式转化 cv2PIL
        img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        img = img.convert('L')
        if remove_bk:
            img = ImageUtils.remove_bk(img)
            img = img.convert('L')
        return img

    # 将 PIL 图片转化成 cv
    @staticmethod
    def img_pil_to_cv(img) -> numpy.ndarray:
        img = cv.cvtColor(numpy.asarray(img), cv.COLOR_RGB2BGR)
        return img


class ClfSheep:
    def __init__(self, pretrain_file):
        # self.model = Net()
        # self.model = self.model.load_state_dict(torch.load(pretrain_file, map_location=torch.device('cpu')))
        self.model = torch.load(pretrain_file, map_location=torch.device('cpu'))
        self.model = self.model.to('cpu')
        self.transform = transforms.Compose([
            transforms.Resize(size=160),  # 将一个边长缩放到160，另一个边按照这个比例进行缩放
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)])

    # 羊毛分类
    def seg_clf(self, clf_img, cfg: Config):
        clf_data = ClfData(clf_img, cfg)
        # clf_data.show_list()
        predict_list = []
        while clf_data.has_item():
            crop_img = clf_data.get_item()
            if self.transform is not None:
                crop_img = self.transform(crop_img)
                crop_img = crop_img.to('cpu')
                crop_img = crop_img.unsqueeze(0)
                predict = self.model(crop_img)
                _, predict = torch.max(predict, 1)
                predict_list.append(predict)
            else:
                return
        return predict_list

    @staticmethod
    def _sheep_vote(candidate: list):
        if candidate[0] != 0 and candidate[1] != 0:
            if candidate[0] * 6 > candidate[1]:
                return 0
        if candidate[3] > sum(candidate) * 0.6:
            return 3
        if candidate[1] != 0 and candidate[2] != 0:
            if candidate[1] > candidate[2]:
                return 1
            else:
                return 2
        return 1

    def sheep_clf(self, clf_img, cfg: Config):
        predict_list = self.seg_clf(clf_img, cfg)
        candidate = [0, 0, 0, 0]
        for predict in predict_list:
            pre = predict.numpy()[0]
            # print(predict, pre)
            if pre == 4:
                continue
            elif int(pre) in [0, 1, 2, 3]:
                candidate[int(pre)] = candidate[int(pre)] + 1

        ans = self._sheep_vote(candidate)
        return ans
