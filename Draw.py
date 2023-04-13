import numpy as np
import cv2
import importlib
Util = importlib.import_module("OpenCV-Python-Utilities")
Calculate = importlib.import_module("OpenCV-Python-Utilities.Calculate")


class Draw:
    @classmethod
    def draw_grid(cls, source: np.ndarray, dst: np.ndarray, col: int, row: int, textarea=False, dst_cont=None) -> None:
        """
        | Draws grid which gets calculated by Calculate.calc_grid()
        |

        :param source: Source image
        :param col: Column count
        :param row: Row count
        :param textarea: If textarea is available
        """
        h, w = source.shape[:2]
        start_x, start_y, start_w, start_h = 0, 0, w, h
        if dst_cont is not None:
            start_x, start_y, start_w, start_h = cv2.boundingRect(dst_cont)
        end_y, end_x = h - ((h - start_y) - start_h), w - ((w - start_x) - start_w)
        textarea_h = 0
        if textarea:
            field_w, checkbox_h, textarea_h = Calculate.Calculate.calc_grid(source, col, row, True, dst_cont=dst_cont)
            field_h = checkbox_h + textarea_h
        else:
            field_w, field_h = Calculate.Calculate.calc_grid(source, col, row, dst_cont=dst_cont)

        # Column lines (top to bottom)
        for c in range(col + 1):
            cv2.line(dst, (start_x + int(field_w * c), start_y), (start_x + int(field_w * c), end_y),
                     (0, 186, 255), 4, cv2.LINE_AA)
        # Row lines (left to right)
        for r in range(row + 1):
            cv2.line(dst, (start_x, start_y + int(field_h * r)), (end_x, start_y + int(field_h * r)),
                     (0, 186, 255), 4, cv2.LINE_AA)
            if textarea:
                cv2.line(dst, (start_x, start_y + int(field_h * r - textarea_h)),
                         (end_x, start_y + int(field_h * r - textarea_h)), (0, 186, 255), 1, cv2.LINE_AA)

    @classmethod
    def draw_cross(cls, dst: np.ndarray, x: int, y: int, w: int, h: int) -> None:
        cv2.line(dst, (int(x + w / 2 - 20), int(y + h / 2 - 20)), (int(x + w / 2 + 20), int(y + h / 2 + 20)),
                 (0, 0, 255), 3, cv2.LINE_AA)
        cv2.line(dst, (int(x + w / 2 - 20), int(y + h / 2 + 20)), (int(x + w / 2 + 20), int(y + h / 2 - 20)),
                 (0, 0, 255), 3, cv2.LINE_AA)

    @classmethod
    def draw_checkmark(cls, dst: np.ndarray, x: int, y: int, w: int, h: int) -> None:
        cv2.line(dst, (int(x + w / 2), int(y + h / 2 + 20)), (int(x + w / 2 + 20), int(y + h / 2 - 20)), (0, 255, 0), 3,
                 cv2.LINE_AA)
        cv2.line(dst, (int(x + w / 2 - 10), int(y + h / 2 + 1)), (int(x + w / 2), int(y + h / 2 + 20)), (0, 255, 0), 3,
                 cv2.LINE_AA)

    @classmethod
    def draw_correctionmark(cls, dst: np.ndarray, x: int, y: int, w: int, h: int) -> None:
        cv2.line(dst, (int(x + w / 2), int(y + h / 2 + 20)), (int(x + w / 2 - 10), int(y + h / 2 - 18)),
                 (0, 186, 255), 3, cv2.LINE_AA)
        cv2.line(dst, (int(x + w / 2), int(y + h / 2 + 20)), (int(x + w / 2 + 10), int(y + h / 2 - 18)),
                 (0, 186, 255), 3, cv2.LINE_AA)
        cv2.line(dst, (int(x + w / 2) - 15, int(y + h / 2 - 6)), (int(x + w / 2 + 15), int(y + h / 2 - 6)),
                 (0, 186, 255), 3, cv2.LINE_AA)

    @classmethod
    def draw_text(cls, dst: np.ndarray, text: str, x_off=0, y_off=0, size=1, thickness=1, color=(0, 0, 0), aa=True,
                  dst_cont=None, text_align_x=Util.CONST.ALIGN_LEFT, text_align_y=Util.CONST.ALIGN_TOP):
        h, w = dst.shape[:2]
        start_x, start_y, start_w, start_h = 0, 0, w, h
        if dst_cont is not None:
            start_x, start_y, start_w, start_h = cv2.boundingRect(dst_cont)

        match text_align_x:
            case Util.CONST.ALIGN_LEFT:
                x = start_x + x_off
            case Util.CONST.ALIGN_RIGHT:
                x = start_x + start_x + x_off
            case Util.CONST.ALIGN_CENTER:
                x = start_x + int(start_w/2) + x_off
            case _:
                print(f"\033[1;31;40mtext_align_x value {text_align_x} not matching any available value")
                assert NotImplementedError
                return NotImplementedError
        match text_align_y:
            case Util.CONST.ALIGN_TOP:
                y = start_y + y_off
            case Util.CONST.ALIGN_BOTTOM:
                y = start_y + start_h + y_off
            case Util.CONST.ALIGN_CENTER:
                y = start_y + int(start_h/2) + y_off
            case _:
                print(f"\033[1;31;40mtext_align_y {text_align_y} not matching any available value")
                assert NotImplementedError
                return NotImplementedError

        if aa:
            cv2.putText(dst, text, (x, y), cv2.QT_FONT_NORMAL, size, color, thickness, cv2.LINE_AA)
        else:
            cv2.putText(dst, text, (x, y), cv2.QT_FONT_NORMAL, size, color, thickness)

    @classmethod
    def draw_line(cls, dst: np.ndarray, p1_x: int, p1_y: int, p2_x: int, p2_y: int, color=(0, 186, 255), cont: list = None, aa=True,
                  thickness=1) -> None:
        start_x, start_y, start_w, start_h = 0, 0, 0, 0
        if cont is not None:
            start_x, start_y, start_w, start_h = cv2.boundingRect(cont)

        if aa:
            cv2.line(dst, (start_x + p1_x, start_y + p1_y), (start_x + p2_x, start_y + p2_y), color, thickness, cv2.LINE_AA)
        else:
            cv2.line(dst, (start_x + p1_x, start_y + p1_y), (start_x + p2_x, start_y + p2_y), color, thickness)

