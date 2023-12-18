import cv2
import os

# выходная папка
img_path = 'Web Images'


# ================================ Основной код ================================

# если нет папки под скрины - сделаем
if not os.path.isdir(img_path):
    os.makedirs(img_path)

# подключаемся к камере
vc = cv2.VideoCapture(0)
image_id = 0

# Если считали первый кадр
if not vc.isOpened():
    print('Проблема к подключении к камере')
    exit()
else:
    rval, last_frame = vc.read()

# настроим окна на вывод
cv2.namedWindow("webcamera")
while rval:
    rval, frame = vc.read()
    cv2.imshow("webcamera", frame)

    # сделаем скрин если нажали на пробел
    key = cv2.waitKey(20)
    if key == 32:
        cv2.imwrite(f'{img_path}/web_{image_id}.png', frame)
        image_id += 1
        print(1)

    # вырубим процесс если нажали на esc
    if key == 27:
        break

cv2.destroyWindow("webcamera")
vc.release()