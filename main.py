import cv2
import numpy as np


# 乒乓球位置识别，加入了指示移动方向的箭头

def empty(a):
    pass


cap = cv2.VideoCapture('./ping_pang_ball.mp4')  # 0对应笔记本自带摄像头

lower = np.array([4, 115, 160])  # 适用于橙色乒乓球4<=h<=32
upper = np.array([32, 255, 255])

ret, img = cap.read()  # 视频的第1帧

imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

imgMask = cv2.inRange(imgHsv, lower, upper)  # 获取遮罩
imgOutput = cv2.bitwise_and(img, img, mask=imgMask)  # 将掩膜和图像逐像素相加

contours, hierarchy = cv2.findContours(imgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 查找轮廓
# CV_RETR_EXTERNAL 只检测最外围轮廓
# CV_CHAIN_APPROX_NONE 保存物体边界上所有连续的轮廓点到contours向量内
imgMask = cv2.cvtColor(imgMask, cv2.COLOR_GRAY2BGR)  # 转换后，后期才能够与原画面拼接，否则与原图维数不同
# 接下来查找包围框，并绘制
x, y, w, h = 0, 0, 0, 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    # print(area)
    if area > 300:
        x, y, w, h = cv2.boundingRect(cnt)
        targetPos_x = int(x + w / 2)
        targetPos_y = int(y + h / 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 以上部分为颜色检测出第一帧的小球包围框区域，以下为根据此区域直方图进行追踪的部分
# 设置初始化的窗口位置
track_window = (x, y, w, h)
# 设置追踪的区域
roi = img[y:y + h, x:x + w]
# roi区域的hsv图像
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# 取值hsv值在(0,60,32)到(180,255,255)之间的部分
mask = cv2.inRange(hsv_roi, lower, upper)
# 计算直方图,参数为 图片(可多)，通道数，蒙板区域，直方图长度，范围
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
# 归一化
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# 设置终止条件，迭代10次或者至少移动1次
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while (1):
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    if ret == True:
        # 计算每一帧的hsv图像
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 计算反向投影
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # 调用meanShift算法在dst中寻找目标窗口，找到后返回目标窗口
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x, y, w, h = track_window
        img2 = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('img2', img2)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

    # 坐标（图像内的）
# cv2.putText(img, "({:0<2d}, {:0<2d})".format(targetPos_x, targetPos_y), (20, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)  # 文字
# imgStack = np.hstack([img, imgOutput])            # 拼接
# cv2.imshow('Horizontal Stacking', imgStack)     # 显示

cv2.destroyAllWindows()
