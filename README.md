# Калибровка камеры
Группа скриптов которые могкт использоваться как для калибровки камеры в дроне и преобразования видеопотока в полете. Либо для нормирования отснятого ролика постфактум. 

----

## image_generate.py
Подключается к интерфейсу камеры и по нажатию кнопки "пробел" генерирует фотографии которые складывает в папку. Для работы скрипта необохдимо распечатать калибровачный кадр по ссылке https://robocraft.ru/files/opencv/acircles_pattern.png

Кнопка "ESC" вырубает цикл. 

---

## calibrate.py
Стучится в папку с кадрами и за счет встроенных методов из OpenCV получает калибровочные матрицы.

---

## using_calibrate.py
Позволяет оценить качество полученых матриц.
