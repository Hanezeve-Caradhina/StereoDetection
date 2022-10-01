import cv2
import time

def shot(folder, pos, frame):
        global counter
        path = folder + pos + "_" + str(counter) + ".jpg"
    
        cv2.imwrite(path, frame)
        print("snapshot saved into: " + path)

def take_photo(interval, counter, folder):
    '''
    interval:拍照间隔，单位秒 e.g.:2为2s
    counter:照片序号（照片名称用）
    folder:保存图片的文件夹
    输出：拍摄左目图片以照片序号为名保存至folder文件夹中
    '''
    INTERVAL = interval # 拍照间隔

    camera = cv2.VideoCapture(0)
    
    # 设置分辨率左右摄像机同一频率，同一设备ID；左右摄像机总分辨率2560x720；分割为两个1280x720
    camera.set(cv2.CAP_PROP_FRAME_WIDTH,2560)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    
    counter = counter
    utc = time.time()
    folder = folder # 拍照文件目录 
    
    while True:
        ret, frame = camera.read()
        print("ret:",ret)
        # 裁剪坐标为[y0:y1, x0:x1]    HEIGHT * WIDTH
        left_frame = frame[0:720, 0:1280]
    
        now = time.time()
        if now - utc >= INTERVAL:
            shot(folder, "left", left_frame)
            counter += 1
            utc = now
    
    camera.release()