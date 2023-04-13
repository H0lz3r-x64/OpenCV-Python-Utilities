import numpy as np
import cv2


class Image:
    @classmethod
    def stackImages(cls, imgArray, scale, lables=None):
        if lables is None:
            lables = []
        sizeW = imgArray[0][0].shape[1]
        sizeH = imgArray[0][0].shape[0]
        rows = len(imgArray)
        cols = len(imgArray[0])
        rowsAvailable = isinstance(imgArray[0], list)
        width = imgArray[0][0].shape[1]
        height = imgArray[0][0].shape[0]
        if rowsAvailable:
            for x in range(0, rows):
                for y in range(0, cols):
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (int(sizeW * scale), int(sizeH * scale)))
                    if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
            imageBlank = np.zeros((height, width, 3), np.uint8)
            hor = [imageBlank] * rows
            hor_con = [imageBlank] * rows
            for x in range(0, rows):
                hor[x] = np.hstack(imgArray[x])
                hor_con[x] = np.concatenate(imgArray[x])
            try:
                ver = np.vstack(hor)
                ver_con = np.concatenate(hor)
            except:
                pass
        else:
            for x in range(0, rows):
                imgArray[x] = cv2.resize(imgArray[x], (int(sizeW * scale), int(sizeH * scale)))
                if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
            hor = np.hstack(imgArray)
            hor_con = np.concatenate(imgArray)
            ver = hor
        if len(lables) != 0:
            eachImgWidth = int(ver.shape[1] / cols)
            eachImgHeight = int(ver.shape[0] / rows)
            for d in range(0, rows):
                for c in range(0, cols):
                    cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                                  (c * eachImgWidth + len(lables[d][c]) * 13 + 27, 30 + eachImgHeight * d),
                                  (255, 255, 255), cv2.FILLED)
                    cv2.putText(ver, lables[d][c], (eachImgWidth * c + 10, eachImgHeight * d + 20),
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
        return ver

    @classmethod
    def warp(cls, dst: np.ndarray, transformation_matrix: list) -> np.ndarray:
        w, h = dst.shape[:2]
        return cv2.warpPerspective(dst, transformation_matrix, (w, h))

    @classmethod
    def get_formated_canny(cls, image: np.ndarray) -> np.ndarray:
        """
        Formats an image to gray, blur, and lastly to canny

        :param image: Source image
        :return: Canny image
        """
        img = image.copy()
        gray = Image.cvt_to_gray(img)
        blur = cv2.GaussianBlur(gray, (5, 5), 1)
        canny = cv2.Canny(blur, 10, 50)
        return canny

    @classmethod
    def size_reduction(cls, canvas: np.ndarray, size_reduction: float) -> np.ndarray:
        """
        | Reduces image size by cutting of a percentage of pixels starting from the image outlines

        :param canvas: Source image
        :param size_reduction: Percentage of pixels that gets cut of
        """
        canvas = Image.cvt_to_gray(canvas)
        h, w = canvas.shape[:2]
        reduce_pixels_h = int(((h / 100) * size_reduction) / 2)
        reduce_pixels_w = int(((w / 100) * size_reduction) / 2)

        x = reduce_pixels_w
        w = w - reduce_pixels_w
        y = reduce_pixels_h
        h = h - reduce_pixels_h
        return canvas[y:h, x:w]

    @classmethod
    def cvt_to_gray(cls, image: np.ndarray) -> np.ndarray:
        image = image.copy()
        if len(image.shape) < 3:
            return image

        channels = image.shape[2]
        match channels:
            case 3:
                try:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                except:
                    try:
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    except:
                        try:
                            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                        except:
                            print("Image format not supported")
                            assert ValueError
        return image

    @classmethod
    def show(cls, img: np.ndarray, winname="test", destroy=False) -> None:
        cv2.imshow(winname, img)
        cv2.waitKey(99999999)
        if destroy:
            cv2.destroyWindow(winname)
