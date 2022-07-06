import os
import glob
import sys 
import random

BASE = 'E:\lake\hui\\'

config = {
    'rate': 0.8
}

if __name__ == '__main__':
    traindata_path = BASE + 'train'
    labels = os.listdir(BASE + 'train')

    train_file = open(BASE + 'train.txt', "w")
    test_file = open(BASE + 'test.txt', "w")

    for index, label in enumerate(labels):
        img_list = glob.glob(os.path.join(traindata_path, label, '*.jpg'))
        random.shuffle(img_list)

        train_list = img_list[:int(config['rate'] * len(img_list))]
        test_list = img_list[(int(config['rate'] * len(img_list)) + 1):]

        for img in train_list:
            train_file.write(img + ' ' + str(index))
            train_file.write('\n')

        for img in test_list:
            test_file.write(img + ' ' + str(index))
            test_file.write('\n')
