import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.transforms.functional as TF
import random


def default_loader(path):
    return Image.open(path)


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        super(MyDataset, self).__init__()

        images = []
        file_list = open(txt, 'r')
        for line in file_list:
            line = line.strip('\n')
            line = line.rstrip('\n')
            if len(line) < 1:
                continue
            words = line.split()
            images.append((words[0], int(words[1])))

        self.images = images
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        img_file_path, label = self.images[index]

        img = self.loader(img_file_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.images)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(10 * 10 * 128, 512)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(512, 5)

    def forward(self, in_put):
        output = self.conv1(in_put)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.pool1(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.pool2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
        output = self.pool3(output)

        output = self.conv4(output)
        output = self.bn4(output)
        output = self.relu4(output)
        output = self.pool4(output)

        output = output.view(-1, 10 * 10 * 128)

        output = self.fc1(output)
        output = self.relu5(output)
        output = self.fc2(output)
        # print(output.shape)
        # return output
        # print('output{},\nF_log{}'.format(output, F.log_softmax(output, dim=1)))
        return F.log_softmax(output, dim=1)  # 内置了cross entropy


# TODO 加入N-flod
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    loss_record = {'train': []}
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        pred = model(data)
        loss = F.nll_loss(pred, target)
        loss_record['train'].append(loss.item())
        # accracy = np.mean((torch.argmax(pred, 1) == torch.argmax(target, 1)).numpy())

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            print('Train Epoch: {}, iteration: {}, Loss: {}'.format(epoch, idx, loss.item()))
            # print('target {} \n pred {}'.format(target, pred.argmax(dim=1)))
   # print('The mean of loss in epoch {} is that {}.'.format(epoch, np.mean(loss_record['train'])))


def test(model, device ,test_loader):
    model.eval()
    total_loss = 0.
    correct = 0.
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            total_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()

    total_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset) * 100.
    print("Test Loss: {}, Accurary:{}".format(total_loss, acc))


class MyRotationTransform:
    """Rotate by one of the given angles"""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


if __name__ == '__main__':
    pathfa = 'E:\lake\hui'
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(size=400, scale=(0, 1)), #对于图片进行裁切
            transforms.RandomResizedCrop(size=(160, 160)),
            # transforms.Resize(size=160),
            # transforms.RandomAdjustSharpness(sharpness_factor=1.2, p=0.5),
            # transforms.RandomRotation(degrees=[90, 180, 270]),
            # MyRotationTransform(angles=[0, 180]),
            # transforms.RandomPerspective(distortion_scale=0.6, p=0.6),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ]),
        'train1': transforms.Compose([
            transforms.FiveCrop(size=(160, 160)),
            # transforms.RandomRotation(degrees=[90, 180, 270]),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ]),
        'test': transforms.Compose([
            # transforms.RandomResizedCrop(size=252, scale=(0.1, 0.9)), #对于图片进行裁切
            # transforms.Resize(size=160),  # 将一个边长缩放到160，另一个边按照这个比例进行缩放
            transforms.RandomResizedCrop(size=(160, 160)),
            # transforms.Resize(size=252),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
    }
    train_data = MyDataset(os.path.join(pathfa, 'train.txt'), transform=data_transforms['train'])
    train_data1 = MyDataset(os.path.join(pathfa, 'train.txt'), transform=data_transforms['train1'])
    test_data = MyDataset(os.path.join(pathfa, 'test.txt'), transform=data_transforms['test'])
    test_data1 = MyDataset(os.path.join(pathfa, 'train.txt'), transform=data_transforms['test'])

    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True, num_workers=4)
    train_loader1 = DataLoader(dataset=train_data1, batch_size=16, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_data, batch_size=16, shuffle=False, num_workers=4)
    test_loader1 = DataLoader(dataset=test_data1, batch_size=16, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lr = 0.01
    momentum = 0.5
    model = Net().to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999),
                                 eps=1e-08, weight_decay=0, amsgrad=False)

    num_epochs = 80
    for epoch in range(num_epochs):
        train(model, device, train_loader, optimizer, epoch)
        train(model, device, train_loader, optimizer, epoch)
        # test(model, device, test_loader)
        test(model, device, test_loader1)

    # sheepModel6 使用裁剪第一次
    # sheepModel7 使用裁剪第二次
    # sheepModel8 重新设计网络
    torch.save(model, 'E:\lake\hui\sheepModel8.pt')