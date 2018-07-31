import cv2
import imutils
import numpy
from imutils import contours
from imutils.perspective import four_point_transform


def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    return rotated_image

def detect_largest_circle(image):
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, int(image.shape[1]/2),
                               param1=50, param2=30, minRadius=int(image.shape[0]/8), maxRadius=int(image.shape[0]/2))
    circles = numpy.uint16(numpy.around(circles))

    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

        cv2.rectangle(image, (i[0]-i[2], i[1]-i[2]), (i[0]+i[2], i[1]+i[2]), (0, 255, 0), 2)

        return image, i[0], i[1], i[2]
        break

    return image, None, None, None

def detect_horizontal_lines(image):

    lines = cv2.HoughLines(image, 1, numpy.pi / 180, 200)
    for rho, theta in lines[0]:
        a = numpy.cos(theta)
        b = numpy.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return image, x1, x2, y1, y2
    return image, None, None, None, None

def run_threshold(image, lower_boundary, upper_boundary, threshold_index=0):
    threshold_list = [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO,
                      cv2.THRESH_TOZERO_INV]

    if threshold_index < 0 or threshold_index > 4:
        raise Exception("Threshold index out of bounds")

    retVal, thresh = cv2.threshold(image, lower_boundary, upper_boundary, threshold_list[threshold_index])

    return thresh


def run_edge_detection(image, detector_index=0):
    if detector_index < 0 or detector_index > 4:
        raise Exception("Detector index out of bounds")

    # Canny detector
    if detector_index == 0:
        # sigma parametr for Canny
        sigma = 0.33
        # compute the median of the single channel pixel intensities
        pixel_median = numpy.median(image)

        # apply automatic Canny edge detection using the computed median
        lower_bound = int(max(0, (1.0 - sigma) * pixel_median))
        upper_bound = int(min(255, (1.0 + sigma) * pixel_median))

        edge_image = cv2.Canny(image, lower_bound, upper_bound)

    # Laplacian edge detection
    elif detector_index == 1:
        edge_image = cv2.Laplacian(image, cv2.CV_64F)

    # Sobel edge detector in x-direction
    elif detector_index == 2:
        edge_image = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)

    # Sobel edge detector in y-direction
    elif detector_index == 3:
        edge_image = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    return edge_image


def find_color(original_img):
    # create lower and upper bounds for red
    red_lower = numpy.array([0, 50, 50], dtype="uint8")
    red_upper = numpy.array([255, 255, 255], dtype="uint8")

    # perform the filtering. mask is another word for filter
    mask = cv2.inRange(original_img, red_lower, red_upper)
    output = cv2.bitwise_and(original_img, original_img, mask=mask)

    return output


def extract_digits_from_image(image):
    # pre-process the image by resizing it, converting it to
    # graycale, blurring it, and computing an edge map
    image = imutils.resize(image, height=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200, 255)

    # find contours in the edge map, then sort them by their
    # size in descending order
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    displayCnt = None

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if the contour has four vertices, then we have found
        # the thermostat display
        if len(approx) == 4:
            displayCnt = approx
            break
    # extract the thermostat display, apply a perspective transform
    # to it
    warped = four_point_transform(gray, displayCnt.reshape(4, 2))
    output = four_point_transform(image, displayCnt.reshape(4, 2))

    # threshold the warped image, then apply a series of morphological
    # operations to cleanup the thresholded image
    thresh = cv2.threshold(warped, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.dilate(thresh, kernel)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # find contours in the thresholded image, then initialize the
    # digit contours lists
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    digitCnts = []

    # loop over the digit area candidates
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # if the contour is sufficiently large, it must be a digit
        if w >= 5 and (h >= 15):
            digitCnts.append(c)

    # sort the contours from left-to-right, then initialize the
    # actual digits themselves
    digitCnts = contours.sort_contours(digitCnts,
                                       method="left-to-right")[0]
    digits = []

    # loop over each of the digits
    for c in digitCnts:
        # extract the digit ROI
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(output, (x, y), (x + w, y + h), (125, 0, 120), 2)

        roi = thresh[y:y + h, x:x + w]

        # compute the width and height of each of the 7 segments
        # we are going to examine
        (roiH, roiW) = roi.shape
        (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
        dHC = int(roiH * 0.05)

        # define the set of 7 segments
        segments = [
            ((0, 0), (w, dH)),  # top
            ((0, 0), (dW, h // 2)),  # top-left
            ((w - dW, 0), (w, h // 2)),  # top-right
            ((0, (h // 2) - dHC), (w, (h // 2) + dHC)),  # center
            ((0, h // 2), (dW, h)),  # bottom-left
            ((w - dW, h // 2), (w, h)),  # bottom-right
            ((0, h - dH), (w, h))  # bottom
        ]
        on = [0] * len(segments)
        # print(on)
        # cv2.imshow("Result", output)
        # cv2.waitKey(0)

        # loop over the segments
        for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
            # extract the segment ROI, count the total number of
            # thresholded pixels in the segment, and then compute
            # the area of the segment
            segROI = roi[yA:yB, xA:xB]
            total = cv2.countNonZero(segROI)
            area = (xB - xA) * (yB - yA)

            # if the total number of non-zero pixels is greater than
            # 50% of the area, mark the segment as "on"
            if total / float(area) > 0.5:
                on[i] = 1

        # lookup the digit and draw it on the image
        try:
            digit = DIGITS_LOOKUP[tuple(on)]
            digits.append(digit)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(output, str(digit), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        except KeyError:
            digit = '?'
            digits.append(digit)

    return output, digits


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