import numpy as np
import cv2 as cv
import glob
import os


img_path = 'Web Images'
calib_file_path = 'Calibration'
show_point_detector = False

# ================================ Известное положение точек ================================

# Критерий поиска углов
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Реальные размеры точки
obj3d = np.zeros((44, 3), np.float32)

# Координата z равна нулю, а координаты x и y - заранее известные числа
a = [0, 36, 72, 108, 144, 180, 216, 252, 288, 324, 360]
b = [0, 72, 144, 216, 36, 108, 180, 252]
for i in range(0, 44):
    obj3d[i] = (a[i // 4], (b[i % 8]), 0)

# ================================ Калибровка ================================

# Массив для хранения 2D и 3D точек
obj_points = []
img_points = []

# Получим кадры из папки
images = glob.glob(f'{img_path}/*.png')

for f in images:
    # Грузим картинку и конвертируем ее в серый
    img = cv.imread(f)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Функция поиска кругов
    ret, corners = cv.findCirclesGrid(
        gray, (4, 11), None, flags=cv.CALIB_CB_ASYMMETRIC_GRID)

    # Если функция нашла обьекты и их центры
    if ret == True:
        obj_points.append(obj3d)

        # В случае круговых сеток
        # функция cornerSubPix() не всегда необходима, поэтому альтернативным методом является
        # corners2 = corners
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners2)

        # Отобразим точки
        if show_point_detector:
            cv.drawChessboardCorners(img, (4, 11), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(0)
cv.destroyAllWindows()

"""Калибровка камеры:
Передача значений известных 3D-точек (obj-точек) и соответствующих пиксельных координат
обнаруженных углов (img-точек)"""
ret, camera_mat, distortion, rotation_vecs, translation_vecs = cv.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None)

# ================================ Запись файлов в папку ================================
if not os.path.isdir(calib_file_path):
    os.makedirs(calib_file_path)

np.savetxt(f'{calib_file_path}/Camera matrix.txt', camera_mat)
np.savetxt(f'{calib_file_path}/Distortion coefficients.txt', distortion)

with open(f'{calib_file_path}/Rotation vector.txt', 'w') as outfile:
    for slice_2d in rotation_vecs:
        np.savetxt(outfile, slice_2d.reshape(1,3))

with open(f'{calib_file_path}/Translation vector.txt', 'w') as outfile:
    for slice_2d in translation_vecs:
        np.savetxt(outfile, slice_2d.reshape(1,3))


# отобразим массивы
print("Error in projection : \n", ret)
print("\nCamera matrix : \n", camera_mat)
print("\nDistortion coefficients : \n", distortion)
print("\nRotation vector : \n", rotation_vecs)
print("\nTranslation vector : \n", translation_vecs)
