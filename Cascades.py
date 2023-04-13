import cv2
import numpy


class Cascades:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier("resources/haarcascade_frontalface_default.xml")

    def find_faces(self, img) -> numpy.ndarray:

        # img = cv2.imread("resources")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(img_gray, 1.1, 4)
        for (x, y, w, h) in faces:
            val = 0
            half = int(val)
            x, y, w, h = x+half, y+half, w-(val+half), h-(val+half)
            cv2.rectangle(img, (x, y), (x + w, y + h), (100, 255, 100), 1)
        return img
