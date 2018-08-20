import cv2
import numpy

from utils import imageAnalysis


def crop_scale_display(image, is_first_frame=False):
    global scale_roi

    rotated_image = imageAnalysis.rotate_image(image, -90)
    rotated_image2 = cv2.fastNlMeansDenoisingColored(rotated_image, None, 10, 10, 7, 21)
    gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 27, -40)

    gray2 = cv2.cvtColor(rotated_image2, cv2.COLOR_BGR2GRAY)
    thresh2 = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 27, -40)

    # extract the scale display ROI from the first frame by finding the largest center circle
    if is_first_frame:
        _, centerX, centerY, radius = imageAnalysis.detect_largest_circle(thresh)
        scale_roi = (centerY - radius, centerX - radius, centerY + radius, centerX + radius)

        if CUTOFF_AT_HORIZONTAL_LINE:
            # crop image and detect borders to further modify ROI
            cropped_image = rotated_image[scale_roi[0]:scale_roi[0] + scale_roi[2],
                            scale_roi[1]:scale_roi[1] + scale_roi[3]]
            cropped_gray = gray[scale_roi[0]:scale_roi[0] + scale_roi[2], scale_roi[1]:scale_roi[1] + scale_roi[3]]
            cropped_thresh = thresh[scale_roi[0]:scale_roi[0] + scale_roi[2], scale_roi[1]:scale_roi[1] + scale_roi[3]]
            borderY = imageAnalysis.detect_horizontal_lines(cropped_thresh, cropped_image)
            print((scale_roi, cropped_thresh.shape, borderY))
            if borderY <= cropped_thresh.shape[0] / 2:
                scale_roi = (centerY - radius + borderY, centerX - radius, centerY + radius + borderY, centerX + radius)
            else:
                scale_roi = (centerY - radius, centerX - radius, borderY, centerX + radius)

    cropped_image = rotated_image[scale_roi[0]:scale_roi[0] + scale_roi[2], scale_roi[1]:scale_roi[1] + scale_roi[3]]
    cropped_gray = gray[scale_roi[0]:scale_roi[0] + scale_roi[2], scale_roi[1]:scale_roi[1] + scale_roi[3]]
    cropped_thresh = thresh[scale_roi[0]:scale_roi[0] + scale_roi[2], scale_roi[1]:scale_roi[1] + scale_roi[3]]

    blurred = cv2.GaussianBlur(cropped_image, (7, 7), 0)
    cropped_gray = cv2.GaussianBlur(cropped_gray, (5, 5), 0)
    # blurred = cv2.medianBlur(cropped_image, 5)
    # kernel = numpy.ones((7, 7), numpy.uint8)
    # processed_image = cv2.erode(blurred, kernel, iterations=1)
    processed_image = blurred

    return processed_image, cropped_gray, cropped_thresh


def crop_scale2(image, is_first_frame=False):
    global scale_roi

    rotated_image = imageAnalysis.rotate_image(image, -90)
    gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.GaussianBlur(gray, (7, 7), 0)

    if is_first_frame:
        blurred_gray = cv2.resize(blurred_gray, (0, 0), fx=2, fy=2)
        dst = preprocess_image(blurred_gray)
        digits_positions = find_digits(dst)
        # get top line for cropping
        top_line = digits_positions[0]
        y0 = top_line[0][1] - 100
        y1 = top_line[1][1]

        scale_roi = [int(y0 / 2), int(y1 / 2), 0, int(blurred_gray.shape[0] / 2)]

    cropped_image = rotated_image[scale_roi[0]:scale_roi[1], scale_roi[2]:scale_roi[3]]
    cropped_gray = blurred_gray[scale_roi[0]:scale_roi[1], scale_roi[2]:scale_roi[3]]

    kernel = numpy.ones((3, 3), numpy.uint8)
    # cropped_image = cv2.erode(cropped_image, kernel, iterations=1)
    # cropped_gray = cv2.erode(cropped_gray, kernel, iterations=1)

    return cropped_image, cropped_gray


def preprocess_image(image):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(6, 6))
    img = clahe.apply(image)
    dst = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 127, 60)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel, iterations=1)
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel, iterations=2)

    cv2.imshow("", dst)

    return dst


def find_digits(image):
    digits_positions = []
    img_array = numpy.sum(image, axis=0)
    horizon_position = helper_extract(img_array, threshold=20)
    img_array = numpy.sum(image, axis=1)
    vertical_position = helper_extract(img_array, threshold=20 * 4)
    # make vertical_position has only one element
    if len(vertical_position) > 1:
        vertical_position = [(vertical_position[0][0], vertical_position[len(vertical_position) - 1][1])]
    for h in horizon_position:
        for v in vertical_position:
            digits_positions.append(list(zip(h, v)))
    assert len(digits_positions) > 0, "Failed to find digits's positions"
    return digits_positions


def helper_extract(one_d_array, threshold=20):
    res = []
    flag = 0
    temp = 0
    for i in range(len(one_d_array)):
        if one_d_array[i] < 8 * 255:
            if flag > threshold:
                start = i - flag
                end = i
                temp = end
                if end - start > 20:
                    res.append((start, end))
            flag = 0
        else:
            flag += 1

    else:
        if flag > threshold:
            start = temp
            end = len(one_d_array)
            if end - start > 50:
                res.append((start, end))
    return res


scale_roi = None
CUTOFF_AT_HORIZONTAL_LINE = False