import cv2
import numpy
import urllib.request, urllib.error, urllib.parse
from utils import imageAnalysis
import imutils
from utils import predictionFastai

from utils.imageAnalysis import DIGITS_LOOKUP

USE_WEBCAM = False
CUTOFF_AT_HORIZONTAL_LINE = False
VISUALIZE = False

if not USE_WEBCAM:
    host = "172.16.50.74:8080"
    hoststream = 'http://' + host + '/shot.jpg'

# Region of interest for scale display
scale_roi = None


def get_img_from_stream():
    if USE_WEBCAM:
        cam = cv2.VideoCapture(0)
        _, img = cam.read()
    else:
        # Use urllib to get the image and convert into a cv2 usable format
        imgResp = urllib.request.urlopen(hoststream)
        imgNp = numpy.array(bytearray(imgResp.read()), dtype=numpy.uint8)
        img = cv2.imdecode(imgNp, -1)
    return img


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


def find_digits2(thresh_image, orig_image):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_DILATE, kernel, iterations=2)
    # thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernel, iterations=2)

    cnts = cv2.findContours(thresh_image.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    digitCnts = []
    cv2.imshow("threshed", thresh_image)

    # loop over the digit area candidates
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # if the contour is sufficiently large, it must be a digit
        if (w >= thresh_image.shape[1] * 0.02 and w <= thresh_image.shape[1] * 0.3) and (h >= 10):
            cv2.rectangle(orig_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            digitCnts.append(c)


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


def recognize_digits_line_method(digits_positions, output_img, input_img):
    digits = []
    # reverse digit list so we read from right to left
    digits_positions = list(reversed(digits_positions))
    for c in digits_positions:
        x0, y0 = c[0]
        x1, y1 = c[1]
        roi = input_img[y0:y1, x0:x1]
        h, w = roi.shape
        suppose_W = max(1, int(h / 1.9))

        # 消除无关符号干扰
        if x1 - x0 < 50 and cv2.countNonZero(roi) / ((y1 - y0) * (x1 - x0)) < 0.2:
            continue

        # 对1的情况单独识别
        if w < suppose_W / 2:
            x0 = max(x0 + w - suppose_W, 0)
            roi = input_img[y0:y1, x0:x1]
            w = roi.shape[1]

        center_y = h // 2
        quater_y_1 = h // 4
        quater_y_3 = quater_y_1 * 3
        center_x = w // 2
        line_width = 5  # line's width
        width = (max(int(w * 0.15), 1) + max(int(h * 0.15), 1)) // 2
        small_delta = int(h / 6.0) // 4
        segments = [
            ((w - 2 * width, quater_y_1 - line_width), (w, quater_y_1 + line_width)),
            ((w - 2 * width, quater_y_3 - line_width), (w, quater_y_3 + line_width)),
            ((center_x - line_width - small_delta, h - 2 * width), (center_x - small_delta + line_width, h)),
            ((0, quater_y_3 - line_width), (2 * width, quater_y_3 + line_width)),
            ((0, quater_y_1 - line_width), (2 * width, quater_y_1 + line_width)),
            ((center_x - line_width, 0), (center_x + line_width, 2 * width)),
            ((center_x - line_width, center_y - line_width), (center_x + line_width, center_y + line_width)),
        ]
        on = [0] * len(segments)

        for (i, ((xa, ya), (xb, yb))) in enumerate(segments):
            seg_roi = roi[ya:yb, xa:xb]
            # plt.imshow(seg_roi, 'gray')
            # plt.show()
            total = cv2.countNonZero(seg_roi)
            area = (xb - xa) * (yb - ya) * 0.9
            # print('prob: ', total / float(area))
            if total / float(area) > 0.25:
                on[i] = 1
        # print('encode: ', on)
        if tuple(on) in DIGITS_LOOKUP.keys():
            digit = DIGITS_LOOKUP[tuple(on)]
        else:
            digit = '*'

        digits.append(digit)

        if cv2.countNonZero(roi[h - int(3 * width / 4):h, w - int(3 * width / 4):w]) / (9 / 16 * width * width) > 0.65:
            digits.append('.')
            cv2.rectangle(output_img,
                          (x0 + w - int(3 * width / 4), y0 + h - int(3 * width / 4)),
                          (x1, y1), (0, 128, 0), 2)
            cv2.putText(output_img, 'dot',
                        (x0 + w - int(3 * width / 4), y0 + h - int(3 * width / 4) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 128, 0), 2)

        cv2.rectangle(output_img, (x0, y0), (x1, y1), (0, 128, 0), 2)
        cv2.putText(output_img, str(digit), (x0 + 3, y0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 128, 0), 2)

    return digits


def recognize_digits_area_method(digits_positions, output_img, input_img):
    digits = []
    for c in digits_positions:
        x0, y0 = c[0]
        x1, y1 = c[1]
        roi = input_img[y0:y1, x0:x1]
        h, w = roi.shape
        suppose_W = max(1, int(h / 1.9))
        # 对1的情况单独识别
        if w < suppose_W / 2:
            x0 = x0 + w - suppose_W
            w = suppose_W
            roi = input_img[y0:y1, x0:x1]
        width = (max(int(w * 0.15), 1) + max(int(h * 0.15), 1)) // 2
        dhc = int(width * 0.8)
        # print('width :', width)
        # print('dhc :', dhc)

        small_delta = int(h / 6.0) // 4
        # print('small_delta : ', small_delta)
        segments = [

            # # version 2
            ((w - width - small_delta, width // 2), (w, (h - dhc) // 2)),
            ((w - width - 2 * small_delta, (h + dhc) // 2), (w - small_delta, h - width // 2)),
            ((width - small_delta, h - width), (w - width - small_delta, h)),
            ((0, (h + dhc) // 2), (width, h - width // 2)),
            ((small_delta, width // 2), (small_delta + width, (h - dhc) // 2)),
            ((small_delta, 0), (w + small_delta, width)),
            ((width - small_delta, (h - dhc) // 2), (w - width - small_delta, (h + dhc) // 2))
        ]

        on = [0] * len(segments)

        for (i, ((xa, ya), (xb, yb))) in enumerate(segments):
            seg_roi = roi[ya:yb, xa:xb]
            # plt.imshow(seg_roi)
            # plt.show()
            total = cv2.countNonZero(seg_roi)
            area = (xb - xa) * (yb - ya) * 0.9
            print(total / float(area))
            if total / float(area) > 0.45:
                on[i] = 1

        # print(on)

        if tuple(on) in DIGITS_LOOKUP.keys():
            digit = DIGITS_LOOKUP[tuple(on)]
        else:
            digit = '*'
        digits.append(digit)
        cv2.rectangle(output_img, (x0, y0), (x1, y1), (0, 128, 0), 2)
        cv2.putText(output_img, str(digit), (x0 - 10, y0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 128, 0), 2)

    return digits


def filter_digits(digit_list):
    filtered_digits = []
    for digits in digit_list:
        if all(x == digits[0] for x in digits):
            continue
        if "*" in digits:
            while digits.count("*") > 0:
                digits.remove("*")
        if "." in digits:
            while digits.count(".") > 0:
                digits.remove(".")

        filtered_digits.append(digits)
    filtered_digits.sort(key=len, reverse=True)
    filtered_digits = numpy.array(filtered_digits)

    return filtered_digits


def main():
    predictionFastai.init_model()
    cap = cv2.VideoCapture('Videos/22.mp4')

    _, firstFrame = cap.read()

    frame, _ = crop_scale2(firstFrame, True)
    digits = []
    try:
        while (cap.isOpened()):
            ret, frame = cap.read()
            alpha = float(1.6)
            frame = cv2.multiply(frame, numpy.array([alpha]))

            preprocessed_image, pre_gray = crop_scale2(frame)

            pre_gray = cv2.resize(pre_gray, (0, 0), fx=2, fy=2)
            dst = preprocess_image(pre_gray)
            digits_positions = find_digits(dst)

            #digits.append(recognize_digits_line_method(digits_positions, pre_gray, dst))
            predictions=predictionFastai.recognize_digits(digits_positions, pre_gray, dst)
            print(predictions)
            digits.append(predictions)

            if VISUALIZE:
                cv2.imshow("Frame", pre_gray)
                cv2.imshow("dst", dst)
                cv2.waitKey(1)
    except Exception as e:
        print(e)

    #filtered_digits = (filter_digits(digits))
    # first, second, third=filtered_digits.max(axis=0)
    #print(filtered_digits)
    # print((first, second, third))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
