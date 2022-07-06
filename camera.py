import cv2
def read_show():
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.avi', fourcc, 25, (640, 480))
    index = 1
    while(cap.isOpened()):
        ret, frame = cap.read()
        #out.write(frame)
        cv2.imshow('自定义', frame)
        k = cv2.waitKey(0) & 0xFF
        if k == ord("q"):
            break
        elif k == ord('s'):
            cv2.imwrite('自定义保存路径+图片命名' + str(index) + ".jpg", frame)
            index += 1
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    read_show()
