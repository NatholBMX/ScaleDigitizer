import cv2
import imutils
import numpy
from imutils import contours
from imutils.perspective import four_point_transform
import math

VISUALIZE = True


def rotate_image(image, angle):
    """
    Rotates image by specified angle
    :param image: Image to rotate
    :param angle: Angle in deg to rotate image
    :return: Rotated image
    """
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    return rotated_image


def detect_largest_circle(image):
    """
    Detects the largest cirlce in the image based on contours
    :param image: Image to detect circle on
    :return: Image with draw circle on; x,y coordinates of centre point, radius of circle
    """
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, int(image.shape[1] / 2),
                               param1=50, param2=30, minRadius=int(image.shape[0] / 8),
                               maxRadius=int(image.shape[0] / 2))

    circles = numpy.uint16(numpy.around(circles))

    for i in circles[0, :]:

        if VISUALIZE:
            # draw the outer circle
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

            cv2.rectangle(image, (i[0] - i[2], i[1] - i[2]), (i[0] + i[2], i[1] + i[2]), (0, 255, 0), 2)

        return image, i[0], i[1], i[2]
        break

    return image, None, None, None


def detect_horizontal_lines(thresh_image, orig_image):
    """
    From the image, detect to outermost horizontal lines
    :param thresh_image: Threshold image for detecting lines
    :param orig_image: Original images to draw lines on
    :return:
    """
    lines = cv2.HoughLinesP(thresh_image, 1, math.pi / 2, 2, None, 30, 1)
    for line in lines:
        x1 = line[0][0]
        x2 = line[0][2]
        y1 = line[0][1]
        y2 = line[0][3]

        # only consider horizontal lines
        if abs(y1 - y2) <= 5:
            # only consider border lines
            lower_boundary = thresh_image.shape[0]
            print((y1, lower_boundary))
            if (y1 < lower_boundary * 0.2 and y1 > lower_boundary * 0.05) or (
                    y1 > lower_boundary * 0.8 and y1 < lower_boundary * 0.95):
                pt1 = (x1, y1)
                pt2 = (x2, y2)

                if VISUALIZE:
                    cv2.line(orig_image, pt1, pt2, (0, 0, 255), 3)
                    cv2.imshow("line", orig_image)
                    cv2.waitKey(0)

                # return y1


DIGITS_LOOKUP = {
    (1, 1, 1, 1, 1, 1, 0): 0,
    (1, 1, 0, 0, 0, 0, 0): 1,
    (1, 0, 1, 1, 0, 1, 1): 2,
    (1, 1, 1, 0, 0, 1, 1): 3,
    (1, 1, 0, 0, 1, 0, 1): 4,
    (0, 1, 1, 0, 1, 1, 1): 5,
    (0, 1, 1, 1, 1, 1, 1): 6,
    (1, 1, 0, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 0, 1, 1, 1): 9,
    (0, 0, 0, 0, 0, 1, 1): '-'
}
