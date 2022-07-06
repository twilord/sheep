from main import Net
import torch
import os
from torchvision import transforms
from PIL import Image
from detection_config import Config


config = Config()
model = torch.load(os.path.join(config.now_cwd, 'pretrain_model/sheepModel5.pt'), map_location=torch.device('cpu'))
model = model.to('cpu')

transform = transforms.Compose([
            transforms.Resize(size=160),  # 将一个边长缩放到160，另一个边按照这个比例进行缩放
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
])

path = 'E:\\lake\\hui\\test'
img = Image.open(os.path.join(path, 'A3-4-2.jpg'))

img = transform(img)
print(img)

img.to('cpu')

prediction = model(img)
print(prediction)