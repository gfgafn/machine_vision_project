import cv2 as cv
import numpy as np

from utils import DrawHelper

ORIGINAL_VIDEO_PATH = './ping_pang_ball.mp4'

BLUE_BRG = (255, 0, 0)
GREEN_BRG = (0, 255, 0)
RED_BRG = (0, 0, 255)

first_frame_of_video = []

# 读取视频
cap = cv.VideoCapture(ORIGINAL_VIDEO_PATH)
if not cap.isOpened():
    print('can not open the video!!!')
else:
    # 获取视频第一帧
    ret, frame = cap.read()
    first_frame_of_video = frame
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
    else:
        cv.namedWindow('Tracking Object', cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.imshow('Tracking Object', frame)

# 鼠标左键按下且移动的起始坐标
start_position_x, start_position_y = -1, -1


def mouse_event_dispatcher(event, x, y, flags, _param=None):
    global start_position_x, start_position_y, first_frame_of_video, tracking_win_x, tracking_win_y, tracking_win_w, \
        tracking_win_h
    _img = first_frame_of_video.copy()

    if event == cv.EVENT_LBUTTONDOWN:
        tracking_win_x = x
        tracking_win_y = y
        start_position_x, start_position_y = x, y
    elif event == cv.EVENT_MOUSEMOVE and (flags & cv.EVENT_FLAG_LBUTTON):
        DrawHelper.draw_rect('Tracking Object', _img, (start_position_x, start_position_y), (x, y), BLUE_BRG, cv.LINE_4)
    elif event == cv.EVENT_LBUTTONUP:
        tracking_win_w = abs(x - tracking_win_x)
        tracking_win_h = abs(y - tracking_win_y)
        DrawHelper.draw_rect('Tracking Object', _img, (tracking_win_x, tracking_win_y),
                             (tracking_win_x + tracking_win_w, tracking_win_y + tracking_win_h), BLUE_BRG,
                             cv.LINE_4)


# 设置鼠标事件回调函数
cv.setMouseCallback('Tracking Object', mouse_event_dispatcher)

tracking_win_x, tracking_win_y, tracking_win_w, tracking_win_h = -1, -1, -1, -1

cv.putText(first_frame_of_video, 'Press the left mouse button to select the trace object', (50, 100),
           cv.FONT_HERSHEY_SIMPLEX, 1.3, RED_BRG, 2, cv.LINE_AA)
cv.putText(first_frame_of_video, 'Press the "C" key to continue after selection', (50, 150),
           cv.FONT_HERSHEY_SIMPLEX, 1.3, RED_BRG, 2, cv.LINE_AA)
cv.imshow('Tracking Object', first_frame_of_video)

# 暂停视频, 等待框选目标, 直到按 'C' 键继续
# 鼠标框选要追踪的目标触发 mouse_event_dispatcher 回调函数, 更新 tracking_win_x/y/w/h
while True:
    if cv.waitKey() == ord('c'):
        break

# 乒乓球颜色的 HSV 阈值
range_lower = np.array([3, 72, 0])
range_upper = np.array([179, 222, 255])

# 设置要追踪的区域
roi = first_frame_of_video[
      tracking_win_y:tracking_win_y + tracking_win_h,
      tracking_win_x:tracking_win_x + tracking_win_w]

hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

mask = cv.inRange(hsv_roi, range_lower, range_upper)

# 计算直方图,参数为 图片(可多)，通道数，蒙板区域，直方图长度，范围
roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
# 归一化
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

# 设置终止条件，迭代 10 次或者至少移动 1 pt
term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
track_window = (tracking_win_x, tracking_win_y, tracking_win_w, tracking_win_h)

# 被标记了追踪框的图像
img_marked = []

while True:
    # 按 ‘Q' 键退出视频播放
    if cv.waitKey(20) & 0xFF == ord('q'):
        break
    else:
        ret, img = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        else:
            # 将一帧 RGB 图像转换为 HSV 图像
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            # 计算反向投影
            dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            # 调用 meanShift 算法在 dst 中寻找目标窗口，找到后返回目标窗口
            ret, track_window = cv.meanShift(dst, track_window, term_crit)
            # 在图像上画出跟踪到的目标窗口
            x, y, w, h = track_window
            img_marked = cv.rectangle(img, (x, y), (x + w, y + h), RED_BRG, 2)
            cv.imshow('Tracking Object', img_marked)

cap.release()

# 按 'Esc' 键退出
cv.putText(img_marked, 'Press the "Esc" key to exit', (50, 150),
           cv.FONT_HERSHEY_SIMPLEX, 1.3, RED_BRG, 2, cv.LINE_AA)
cv.imshow('Tracking Object', img_marked)
while True:
    if cv.waitKey() == 27:
        break

cv.destroyAllWindows()
