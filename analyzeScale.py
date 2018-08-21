"""
For videos: the glare should be NOT on the digits but above/below it
"""

import cv2
import numpy
import urllib.request, urllib.error, urllib.parse
import imutils
import imutils.perspective
from operator import itemgetter
from skimage import img_as_ubyte
from skimage.filters import threshold_local
import os
import pickle
import sys
import argparse

from utils.deprecated_methods import crop_scale2, preprocess_image, find_digits
from utils.imageAnalysis import DIGITS_LOOKUP

USE_WEBCAM = False
VISUALIZE = False

VIDEOS_DIR_PATH = "./videos2/"
DATABASE_PATH = "./database/data"

if not USE_WEBCAM:
    host = "172.16.50.74:8080"
    hoststream = 'http://' + host + '/shot.jpg'


# Region of interest for scale display


def get_img_from_stream():
    """
    Gets the image from an input stream, either webcam or IP Cam app
    :return: Image from stream
    """
    if USE_WEBCAM:
        cam = cv2.VideoCapture(0)
        _, img = cam.read()
    else:
        # Use urllib to get the image and convert into a cv2 usable format
        imgResp = urllib.request.urlopen(hoststream)
        imgNp = numpy.array(bytearray(imgResp.read()), dtype=numpy.uint8)
        img = cv2.imdecode(imgNp, -1)
    return img


def get_biggest_difference(lines):
    """
    Get the biggest distance between lines
    :param lines: list of lines
    :return: Lines with the biggest difference in distance
    """
    # there's probably an OpenCV function for this
    xs = [i[0] for i in lines]
    biggest = 0
    biggestid = 0
    for i in range(1, len(xs)):
        dif = xs[i] - xs[i - 1]
        if dif > biggest:
            biggest = dif
            biggestid = i

    return biggestid - 1, biggestid


def contours_to_points(contour, padding=20):
    """
    Get vertex points from rectangular contour
    :param contour: Rectanular input contour
    :return: Vertex points with extra padding
    """
    # this is NOT how you are supposed to do this
    pts = str(contour).split("\n")
    pts = sorted(
        [[int(i) for i in pt.replace("[", "").replace("]", "").split(" ") if i != ""] for pt in pts if pt != ""],
        key=itemgetter(1))
    return [[pts[0][0], pts[0][1] + padding], [pts[-1][0], pts[-1][1] + padding]]


def preprocess_image(image):
    """
    Does all the pre-processing for the image
    :param image:
    :return:
    """
    # blur and convert to grayscale
    gray = cv2.GaussianBlur(image, (11, 11), 0)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    # canny edge detection
    canny = cv2.Canny(gray, 3, 25)

    # hough lines
    lines = cv2.HoughLines(canny, 1, numpy.pi / 180, 150)
    linecoords = []
    for line in lines:
        for rho, theta in line:
            a = numpy.cos(theta)
            b = numpy.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            linecoords.append([x1, y1, x2, y2])

    # find the biggest space between lines
    lines = sorted(linecoords, key=itemgetter(0))
    biggest_difference = get_biggest_difference(lines)

    y, x, _ = image.shape

    #   p1------p2
    #   |        |
    #   |        |
    #   p3------p4

    p1 = (lines[biggest_difference[0]][0], lines[biggest_difference[0]][1])
    p3 = (lines[biggest_difference[0]][2], lines[biggest_difference[0]][3])
    p2 = (lines[biggest_difference[1]][0], lines[biggest_difference[1]][1])
    p4 = (lines[biggest_difference[1]][2], lines[biggest_difference[1]][3])

    orig = image.copy()
    lines = image.copy()
    lines2 = image.copy()

    # draw lines with biggesr space on lines with green lines
    lines = cv2.line(lines, p1, p3, (0, 255, 0), 1)
    lines = cv2.line(lines, p2, p4, (0, 255, 0), 1)

    # purely to see what's going on, also draw on a normal image
    lines2 = cv2.line(lines2, p1, p3, (0, 255, 0), 2)
    lines2 = cv2.line(lines2, p2, p4, (0, 255, 0), 2)

    # threshold lines image and find contours
    lines = cv2.cvtColor(lines, cv2.COLOR_BGR2HSV)
    lines = cv2.inRange(lines, (0, 255, 255), (255, 255, 255))

    # draw black bars
    black_bar = 20
    image = cv2.copyMakeBorder(image, black_bar, black_bar, 0, 0, cv2.BORDER_CONSTANT)
    orig = cv2.copyMakeBorder(orig, black_bar, black_bar, 0, 0, cv2.BORDER_CONSTANT)

    contours = cv2.findContours(lines.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    # only proceed if two contours are found
    if len(contours) == 2:
        # convert contours into strings and append them to a list
        cntpts = []
        for c in contours:
            pts = contours_to_points(c)
            cntpts.append(pts)

        # convert list into one big contour
        allcontours = numpy.array(cntpts).reshape((-1, 1, 2)).astype(numpy.int32)

        # draw find bounding box
        rect = cv2.minAreaRect(allcontours)
        box = cv2.boxPoints(rect)
        box = numpy.int0(box)
        cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

        # perspective transform
        cropped = imutils.perspective.four_point_transform(orig, box.reshape(4, 2))

        # rotate image
        py, px, _ = cropped.shape
        if py > px:
            cropped = imutils.rotate_bound(cropped, 90)
            py, px, _ = cropped.shape

        cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        # use skimage to do adaptive thresholding
        adaptive_thresh = threshold_local(cropped_gray, block_size=89, offset=10)
        cropped_gray = cropped_gray > adaptive_thresh

        # convert back to cv2
        cropped_gray = img_as_ubyte(cropped_gray)
        # blur a bit to get rid of small specks
        cropped_gray = cv2.medianBlur(cropped_gray, 5)
        cropped_gray = cv2.GaussianBlur(cropped_gray, (3, 3), 2)

    cropped_gray = cv2.bitwise_not(cropped_gray)
    return cropped, cropped_gray


def find_digits(thresh_image, orig_image):
    """
    Find the digits inside threshold image
    :param thresh_image: Threshold image for finding digits
    :param orig_image: Original image to draw bounding boxes on
    :return: Digit positions top left and bottom right points
    """

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
    thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_DILATE, kernel, iterations=1)

    contours = cv2.findContours(thresh_image.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if imutils.is_cv2() else contours[1]
    digit_countours = []
    # cv2.imshow("threshed", thresh_image)

    # loop over the digit area candidates
    digits_positions = []
    for contour in contours:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # if the contour is sufficiently large, it must be a digit
        if (w >= thresh_image.shape[1] * 0.02 and w <= thresh_image.shape[1] * 0.4) and (
                h >= thresh_image.shape[0] * 0.2 and h <= thresh_image.shape[0] * 0.6):
            cv2.rectangle(orig_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            digit_countours.append(contour)
            x0 = x
            y0 = y
            x1 = x + w
            y1 = y + h

            digits_positions.append(((x0, y0), (x1, y1)))

    return digits_positions


def recognize_digits_line_method(digits_positions, output_img, input_img):
    """
    Recognize the seven segment digitis with a line method
    :param digits_positions: Positions of digits founds
    :param output_img: Output image to visualize
    :param input_img: Input image to detect images from (threshold image)
    :return: Recognized digits in a list
    """

    digits = []

    for c in digits_positions:
        x0, y0 = c[0]
        x1, y1 = c[1]
        roi = input_img[y0:y1, x0:x1]
        digit_height, digit_width = roi.shape
        suppose_W = max(1, int(digit_height / 1.9))

        if x1 - x0 < 50 and cv2.countNonZero(roi) / ((y1 - y0) * (x1 - x0)) < 0.2:
            continue

        if digit_width < suppose_W / 2:
            x0 = max(x0 + digit_width - suppose_W, 0)
            roi = input_img[y0:y1, x0:x1]
            digit_width = roi.shape[1]

        center_y = digit_height // 2
        quater_y_1 = digit_height // 4
        quater_y_3 = quater_y_1 * 3
        center_x = digit_width // 2
        line_width = 5  # line's width
        width = (max(int(digit_width * 0.15), 1) + max(int(digit_height * 0.15), 1)) // 2
        small_delta = int(digit_height / 6.0) // 4
        segments = [
            ((digit_width - 2 * width, quater_y_1 - line_width), (digit_width, quater_y_1 + line_width)),
            ((digit_width - 2 * width, quater_y_3 - line_width), (digit_width, quater_y_3 + line_width)),
            ((center_x - line_width - small_delta, digit_height - 2 * width),
             (center_x - small_delta + line_width, digit_height)),
            ((0, quater_y_3 - line_width), (2 * width, quater_y_3 + line_width)),
            ((0, quater_y_1 - line_width), (2 * width, quater_y_1 + line_width)),
            ((center_x - line_width, 0), (center_x + line_width, 2 * width)),
            ((center_x - line_width, center_y - line_width), (center_x + line_width, center_y + line_width)),
        ]
        on = [0] * len(segments)

        for (i, ((xa, ya), (xb, yb))) in enumerate(segments):
            seg_roi = roi[ya:yb, xa:xb]
            total = cv2.countNonZero(seg_roi)
            area = (xb - xa) * (yb - ya) * 0.9
            if total / float(area) > 0.25:
                on[i] = 1
        if tuple(on) in DIGITS_LOOKUP.keys():
            digit = DIGITS_LOOKUP[tuple(on)]
        else:
            digit = '*'

        digits.append(digit)

        if cv2.countNonZero(
                roi[digit_height - int(3 * width / 4):digit_height, digit_width - int(3 * width / 4):digit_width]) / (
                9 / 16 * width * width) > 0.65:
            digits.append('.')
            cv2.rectangle(output_img,
                          (x0 + digit_width - int(3 * width / 4), y0 + digit_height - int(3 * width / 4)),
                          (x1, y1), (0, 128, 0), 2)
            cv2.putText(output_img, 'dot',
                        (x0 + digit_width - int(3 * width / 4), y0 + digit_height - int(3 * width / 4) - 10),
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

        if w < suppose_W / 2:
            x0 = x0 + w - suppose_W
            w = suppose_W
            roi = input_img[y0:y1, x0:x1]
        width = (max(int(w * 0.15), 1) + max(int(h * 0.15), 1)) // 2
        dhc = int(width * 0.8)


        small_delta = int(h / 6.0) // 4

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

            total = cv2.countNonZero(seg_roi)
            area = (xb - xa) * (yb - ya) * 0.9
            print(total / float(area))
            if total / float(area) > 0.45:
                on[i] = 1

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
        if len(digits) != 3:
            continue

        filtered_digits.append(digits)
    filtered_digits.sort(key=len, reverse=True)
    filtered_digits = numpy.array(filtered_digits)

    return filtered_digits


def get_most_frequent(digit_list):
    uniques, counts = numpy.unique(digit_list, return_counts=True, axis=0)
    max_index = numpy.argmax(counts)

    return uniques[max_index]


def get_average_from_array(array):
    sum = 0
    for digits in array:
        digits = reversed(digits)
        current_digit = 0
        for i, value in enumerate(digits):
            current_digit = current_digit + value * (10 ** i)
        sum += current_digit

    average = sum / len(array)

    return average / 10


def save_to_database(average, database_path):
    if not os.path.isfile(database_path):
        pickle.dump(average, open(database_path, "wb"))
        return

    last_value = read_all_from_database(database_path)[-1]
    relative_change = average - last_value

    pickle.dump(average, open(database_path, "ab"))

    return relative_change


def read_all_from_database(database_path):
    results = []
    with open(database_path, "rb") as file_stream:
        while True:
            try:
                results.append(pickle.load(file_stream))
            except EOFError:
                break
    return results


def parse_arguments(args):
    parser = argparse.ArgumentParser(description='Run the scale digitizer script')
    parser.add_argument('-video_path', help="Path to folder with videos to process", default=VIDEOS_DIR_PATH)
    parser.add_argument('-database_name', help="Name of database", default=DATABASE_PATH)

    arguments = parser.parse_args(args)

    return arguments


def main2():
    cap = cv2.VideoCapture('Videos/12.mp4')

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

            digits.append(recognize_digits_line_method(digits_positions, pre_gray, dst))

            cv2.imshow("Frame", pre_gray)
            cv2.imshow("dst", dst)
            cv2.waitKey(1)
    except Exception as e:
        print(e)

    filtered_digits = (filter_digits(digits))
    # first, second, third=filtered_digits.max(axis=0)
    print(filtered_digits)
    # print((first, second, third))

    cap.release()
    cv2.destroyAllWindows()


def main(args):
    args=parse_arguments(args)

    video_list = os.listdir(args.video_path)
    digits_for_week = []

    for video in video_list:
        cap = cv2.VideoCapture(args.video_path + video)
        digits = []
        try:
            while (cap.isOpened()):
                ret, frame = cap.read()
                cropped, cropped_gray = preprocess_image(frame)

                digit_pos = find_digits(cropped_gray, cropped)

                digits.append(recognize_digits_line_method(digit_pos, cropped, cropped_gray))

                if VISUALIZE:
                    cv2.imshow("1", cropped)
                    cv2.imshow("2", cropped_gray)

                    cv2.waitKey(1)
        except Exception as e:
            print(e)
        filtered_digits = filter_digits(digits)
        if len(filtered_digits) > 0:
            digits_for_week.append(get_most_frequent(filtered_digits))

    average = get_average_from_array(digits_for_week)
    relative_change = save_to_database(average, args.database_name)
    print(average)
    print(relative_change)


if __name__ == '__main__':
    main(sys.argv[1:])
