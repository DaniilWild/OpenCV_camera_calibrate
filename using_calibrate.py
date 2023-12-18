import numpy as np
import cv2

video_path = '/Users/dtsarck/Desktop/copters/crop_first.mp4'
calib_file_path = 'Calibration'
using_roi = False      # обрезать ли зону искревления камеры
using_center = False   # оставить ли только центр (полезно для больштх кадров)

# параметры цента
y_split, x_split = 800, 800
h_split, w_split = 2000, 1800

# ================================ Посчитаем матрицу перехода ================================

distortion = np.loadtxt(f'{calib_file_path}/Distortion coefficients.txt')
camera_mat = np.loadtxt(f'{calib_file_path}/Camera matrix.txt')

# поклюсимся к видео источнику
vc = cv2.VideoCapture(video_path)
rval, last_frame = vc.read()

# параметры новой матрицы
h, w = last_frame.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_mat, distortion, (w,h), 1, (w, h))

# нивилируем погрешность
mapx, mapy = cv2.initUndistortRectifyMap(camera_mat, distortion, None, newcameramtx, (w, h), 5)

# ================================ Цикл с прогоном ролика ================================

while rval:
    # получаем кадр и
    rval, frame = vc.read()
    if using_center:
        frame = frame[y_split: y_split + h_split, x_split: x_split + w_split]

    # внесем изменения
    dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    if using_roi:
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]

    # вывод всех фреймов на экран
    cv2.imshow("webcamera", frame)
    cv2.imshow('calibresult', dst)

    # вырубим процесс если нажали на esc
    key = cv2.waitKey(20)
    if key == 27:
        break

# вырубим все окна
cv2.destroyWindow("webcamera")
cv2.destroyWindow("calibresult")