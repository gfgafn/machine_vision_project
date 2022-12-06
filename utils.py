import cv2 as cv


def empty(_a):
    pass


class DrawHelper:
    @staticmethod
    def draw_rect(window_name, img, pt1, pt2, color, thickness=None, line_type=None, shift=None):
        cv.rectangle(img, pt1, pt2, color, thickness, line_type, shift)
        cv.imshow(window_name, img)
