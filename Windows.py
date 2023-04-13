import numpy as np
import cv2
import importlib
Image = importlib.import_module("OpenCV-Python-Utilities.Image")


class Windows:
    @classmethod
    def start_screen(cls, canvas, winname) -> None:
        """
        Simple start screen with text

        :param canvas: Background image
        :param winname: Window name
        """
        x, y = (int(canvas.shape[1] / 2 - 200), 30)
        cv2.rectangle(canvas, (x - 10, y - 40), (x + 458, y + 20), (10, 10, 10), cv2.FILLED)
        cv2.putText(canvas, "Press Space to scan image", (x, y), cv2.QT_FONT_NORMAL, 1, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.imshow(winname, canvas)

    @classmethod
    def Image_Masking_Window(cls, window_name: str, img):
        HSV_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Create Window
        cv2.namedWindow(window_name)
        cv2.resizeWindow(window_name, 640, 320)

        # Create Trackbars
        cv2.createTrackbar("Hue Min", window_name, 0, 179, lambda x: x)
        cv2.createTrackbar("Hue Max", window_name, 179, 179, lambda x: x)
        cv2.createTrackbar("Sat Min", window_name, 0, 255, lambda x: x)
        cv2.createTrackbar("Sat Max", window_name, 255, 255, lambda x: x)
        cv2.createTrackbar("Val Min", window_name, 0, 255, lambda x: x)
        cv2.createTrackbar("Val Max", window_name, 255, 255, lambda x: x)

        while True:
            # Get trackbar Values
            H_min = cv2.getTrackbarPos("Hue Min", window_name)
            H_max = cv2.getTrackbarPos("Hue Max", window_name)
            S_min = cv2.getTrackbarPos("Sat Min", window_name)
            S_max = cv2.getTrackbarPos("Sat Max", window_name)
            V_min = cv2.getTrackbarPos("Val Min", window_name)
            V_max = cv2.getTrackbarPos("Val Max", window_name)

            # Mask
            lower = np.array([H_min, S_min, V_min])
            upper = np.array([H_max, S_max, V_max])
            mask = cv2.inRange(HSV_img, lower, upper)

            # Output
            result = cv2.bitwise_and(img, img, mask=mask)

            display_mask = np.copy(mask)
            cv2.putText(display_mask, "mask", (10, 40), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 0, 0), 3)
            display_img = np.copy(img)
            cv2.putText(display_img, "source", (10, 40), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 0, 0), 3)
            display_HSV = np.copy(HSV_img)
            cv2.putText(display_HSV, "HSV", (10, 40), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 0, 0), 3)
            display_result = np.copy(result)
            cv2.putText(display_result, "result", (10, 40), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 0, 0), 3)

            img_stack = Image.Image.stackImages(([display_img, display_HSV], [display_mask, display_result]), 0.6)
            cv2.imshow("Result", img_stack)
            cv2.waitKey(1)