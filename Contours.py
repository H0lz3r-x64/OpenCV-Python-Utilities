import numpy as np
import cv2
from numpy import ndarray

from Util import Windows
from Util import Image
from Util import Calculate


class Contours:
    @classmethod
    def contour_list(cls, img_canny, shape="rectangle", minarea=1, ignore_shape=False) -> list:
        """
        Finds contours in image; preferably an image returned by Image.get_formated_canny()

        :param img_canny: Source image
        :param shape: The shape the contours should have e.g. rectangle
        :param minarea: The minimum area each contour should have
        :return: List containing the found contours matching the shape. In descending order, from biggest to smallest
        """
        # Get the corner count for the selected shape
        if shape == "rectangle":
            # Rectangle (4 Corners)
            corners = 4
        elif shape == "triangle":
            # Triangle (3 Corners)
            corners = 3
        else:
            raise ValueError(f"Invalid shape: \"{shape}\"")
        contour_list = []
        # Find contours and loop through them
        contours = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for item in contours:
            area = cv2.contourArea(item)
            # Check if area is larger than x pixel
            if area > minarea:
                if not ignore_shape:
                    peri = cv2.arcLength(item, True)
                    approx = cv2.approxPolyDP(item, 0.02 * peri, True)
                    # Append the items of the selected shape
                    if len(approx) == corners:
                        contour_list.append(item)
                else:
                    contour_list.append(item)
        # Sort the lists in descending order
        contour_list = sorted(contour_list, key=cv2.contourArea, reverse=True)
        return contour_list

    @classmethod
    def cvt2corner_points(cls, contour_list) -> list:
        """
        Converts list of contours in list of corner points

        :param contour_list: List containing contours returned by cv2.findContours()
        :return: None or a list containing the approximated corner points for each contour in contour_list
        """

        new_contour_list = []
        if len(contour_list) > 1:
            for contour in contour_list:  # iterate through all found contours
                contour = Contours.get_corner_points(contour)  # get corner points
                new_contour_list.append(contour)
            return new_contour_list
        return []

    @classmethod
    def get_corner_points(cls, cont) -> list:
        """
        Gets the corner points of a contour

        :param cont: Contour
        :return: List containing the approximated corner points of the contour
        """
        peri = cv2.arcLength(cont, True)
        approx = cv2.approxPolyDP(cont, 0.02 * peri, True)
        return approx

    @classmethod
    def reorder_points(cls, points) -> ndarray:
        """
        Reorderes list of corner points to make it compatible with cv2.getPerspectiveTransform

        :param points: List containing the points
        :return: List containing the reordered points
        """
        points = points.reshape((4, 2))
        add = points.sum(1)
        new_points = np.zeros((4, 1, 2), np.uint32)
        new_points[0] = points[np.argmin(add)]  # 0 & 0
        new_points[3] = points[np.argmax(add)]  # w & h
        difference = np.diff(points, axis=1)
        new_points[1] = points[np.argmin(difference)]  # w & 0
        new_points[2] = points[np.argmax(difference)]  # 0 & h
        return new_points

    @classmethod
    def draw_contours(cls, dst, contour_corner_points_list, draw_count) -> None:
        """
        Draws contours onto the canvas

        :param dst: Canvas on which gets drawn on
        :param contour_corner_points_list: List containing corner points returned by cont_get_corner_points()
        :param draw_count: Count of how many contours are to be drawn. (from the beginning of the list)
        """
        for i, contour in enumerate(contour_corner_points_list):
            if i == draw_count:
                return
            if contour.size != 0:
                for j in range(len(contour) - 1):
                    cv2.line(dst, contour[j][0], contour[j + 1][0],
                             (0, 186, 255), 6, cv2.LINE_AA)
                # Last edge to first
                cv2.line(dst, contour[0][0], contour[len(contour) - 1][0],
                         (0, 186, 255), 6, cv2.LINE_AA)

    @classmethod
    def cutout_contour(cls, img, contour: list) -> ndarray:
        """
        Creates a new image cutout from a contour

        :param img: Image from which should be cutout
        :param contour: Contour which gets cutout from img
        :return: New image of cutout area
        """
        x, y, w, h = cv2.boundingRect(contour)
        return img[y:y + h, x:x + w]

    @classmethod
    def contour_content_transfer(cls, target_contour: list,  target_source: ndarray, transfer_source: ndarray,
                                 dst: ndarray = None, channel=0,  channel_value=255) -> ndarray:
        if dst is None:
            mask = np.zeros_like(target_source)
            color = [0, 0, 0]
            color[channel] = channel_value
            mask = cv2.drawContours(mask, target_contour, 0, color, thickness=cv2.FILLED)
            h, w = mask.shape[:2]
            for row in range(h):
                for col in range(w):
                    if mask[row, col, channel] == channel_value:
                        for c in range(3):
                            target_source[row, col, c] = transfer_source[row, col, c]
            return target_source
        else:
            mask = np.zeros_like(dst)
            color = [0, 0, 0]
            color[channel] = channel_value
            mask = cv2.drawContours(mask, target_contour, 0, color, thickness=cv2.FILLED)
            h, w = mask.shape[:2]
            for row in range(h):
                for col in range(w):
                    if mask[row, col, channel] == channel_value:  # mask pixels from transfer_source
                        for c in range(3):
                            dst[row, col, c] = transfer_source[row, col, c]
                    else:
                        for c in range(3):  # other pixels from target_source
                            dst[row, col, c] = target_source[row, col, c]
            return dst
