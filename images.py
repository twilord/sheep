from rembg import remove
import os
from PIL import Image
import glob
import PIL
import onnxruntime
import cv


cv.NamedWindow("Camera 1")
cv.NamedWindow("Camera 2")
video1 = cv.CaptureFromCAM(0)
cv.SetCaptureProperty(video1, cv.CV_CAP_PROP_FRAME_WIDTH, 800)
cv.SetCaptureProperty(video1, cv.CV_CAP_PROP_FRAME_HEIGHT, 600)

video2 = cv.CaptureFromCAM(1)
cv.SetCaptureProperty(video2, cv.CV_CAP_PROP_FRAME_WIDTH, 800)
cv.SetCaptureProperty(video2, cv.CV_CAP_PROP_FRAME_HEIGHT, 600)

loop = True
while(loop == True):
    frame1 = cv.QueryFrame(video1)
    frame2 = cv.QueryFrame(video2)
    cv.ShowImage("Camera 1", frame1)
    cv.ShowImage("Camera 2", frame2)
    char = cv.WaitKey(99)
    if (char == 27):
        loop = False

cv.DestroyWindow("Camera 1")
cv.DestroyWindow("Camera 2")

if False:
    path_images = 'E:\lake\origin\D\\'
    # path_images = 'C:\\Users\\twilord\\Pictures\\Saved Pictures\\'

    li = glob.glob(os.path.join(path_images + "*.jpg"))


    for index, img in enumerate(li):
        pic = remove(Image.open(img))
        pic = pic.convert('RGB')
        pic.save('{}D-{}.jpg'.format(path_images, index))