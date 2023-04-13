import numpy as np
import cv2

from Util import Windows
from Util import Contours
from Util import Image


class Calculate:
    @staticmethod
    def calc_grid(canvas, col, row, textarea=False, dst_cont=None) -> tuple:
        # TODO dynamically calculate textarea height from line count
        """
        | Calculates every field's width, checkbox height and textarea height

        :param canvas: Area of which the grid gets calculated
        :param col: Columns of grid
        :param row: Rows of grid
        :param textarea: True if textarea should get calculated
        :return: -> Tuple: (field_w, checkbox_h, textarea_h)
                 -> Tuple: (field_w, field_h)
        """
        h, w = canvas.shape[:2]
        start_x, start_y, start_w, start_h = 0, 0, w, h
        if dst_cont is not None:
            start_x, start_y, start_w, start_h = cv2.boundingRect(dst_cont)
        end_y, end_x = (h - start_y) - ((h - start_y) - start_h), (w - start_x) - ((w - start_x) - start_w)

        # h, w = canvas.shape[:2]
        if textarea:
            field_w = end_x / col
            field_h = end_y / row
            textarea_h = field_h / 4
            checkbox_h = field_h - textarea_h
            return int(field_w), int(checkbox_h), int(textarea_h)
        else:
            field_w = end_x / col
            field_h = end_y / row
            return int(field_w), int(field_h)

    @staticmethod
    def filled_pixels_percentage(canvas: np.ndarray, rounding_digits=2, display=False, print_result=False) -> float:
        """
        | Calculates filled pixels in image
        |
        | • Converts image to GREY
        | • Puts a threshold (THRESH_BINARY_INV) on the image to achieve, empty pixels
        | ⠀ for white, and black pixels for filled pixels.
        |
        | If needed standard threshold configuration can be adjusted by min_thresh, max_thresh parameters.

        :param canvas: Source image
        :param rounding_digits: Number of digits result gets rounded to
        :param display: If True; displays threshold image
        :param print_result: If True; prints percentage
        :param min_thresh: Threshold thresh parameter
        :param max_thresh: Threshold max_val parameter
        :return: Float percentage of filled pixels
        """
        canvas = Image.Image.cvt_to_gray(canvas)
        h, w = canvas.shape[:2]
        pixels = h*w

        # Adaptive
        thresh = cv2.adaptiveThreshold(canvas, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 71, 2)
        filled = np.count_nonzero(thresh)  # TODO this

        percent = round((filled / pixels) * 100, rounding_digits)
        if print_result:
            print(f"{percent} %")
        if display:
            Image.Image.show(thresh, "get_filled_percentage threshold", True)
        return percent

    @classmethod
    def transformation_matrix(cls, image: np.ndarray, dst_corner_points: list) -> list:
        w, h = image.shape[:2]
        reordered = Contours.Contours.reorder_points(dst_corner_points)
        pts1 = np.float32(reordered)
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        transf_matrix = cv2.getPerspectiveTransform(pts1, pts2)
        return transf_matrix

    @classmethod
    def reverse_transformation_matrix(cls, image: np.ndarray, dst_corner_points: list) -> list:
        w, h = image.shape[:2]
        reordered = Contours.Contours.reorder_points(dst_corner_points)
        pts1 = np.float32(reordered)
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        transf_matrix = cv2.getPerspectiveTransform(pts2, pts1)
        return transf_matrix

