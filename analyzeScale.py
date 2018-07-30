# import the necessary packages
import cv2
import numpy
import urllib.request, urllib.error, urllib.parse
from utils import imageAnalysis
import imutils
from keras.models import load_model

USE_WEBCAM = False

if not USE_WEBCAM:
    host = "172.16.50.74:8080"
    hoststream = 'http://' + host + '/shot.jpg'

# Region of interest for scale display
scale_roi = None


# model = load_model("./weights/model.h5")


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


def preprocess_image(image, is_first_frame=False):
    global scale_roi

    rotated_image = imageAnalysis.rotate_image(image, -90)
    gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
    thresh = imageAnalysis.run_threshold(gray, 50, 120, 1)

    # extract the scale display ROI from the first frame
    if is_first_frame:
        _, centerX, centerY, radius = imageAnalysis.detect_largest_circle(thresh)
        scale_roi = (centerY - radius, centerX - radius, centerY + radius, centerX + radius)

    cropped_image = rotated_image[scale_roi[0]:scale_roi[0] + scale_roi[2], scale_roi[1]:scale_roi[1] + scale_roi[3]]
    cropped_gray = gray[scale_roi[0]:scale_roi[0] + scale_roi[2], scale_roi[1]:scale_roi[1] + scale_roi[3]]
    cropped_thresh = thresh[scale_roi[0]:scale_roi[0] + scale_roi[2], scale_roi[1]:scale_roi[1] + scale_roi[3]]
    # blurred = cv2.GaussianBlur(cropped_image, (7, 7), 0)
    blurred = cv2.medianBlur(cropped_image, 5)
    kernel = numpy.ones((7, 7), numpy.uint8)
    processed_image = cv2.erode(blurred, kernel, iterations=1)

    return processed_image, cropped_gray, cropped_thresh


def find_digits(image):
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if imutils.is_cv2() else contours[1]
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)
    digit_contours = []

    # loop over the contours
    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if the contour has four vertices, then we have found a possible candidate
        x, y, w, h = cv2.boundingRect(c)
        # first check if the size of the contours fits
        if h >= image.shape[0] / 5 and h < image.shape[0] / 3 and w < image.shape[1] / 6:
            digit_contours.append(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)

    return image, digit_contours


def recognize_digit(image, contour):
    digits = []
    (x, y, w, h) = cv2.boundingRect(contour)
    roi = image[y:y + h, x:x + w]

    (roiH, roiW) = roi.shape
    (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
    dHC = int(roiH * 0.05)

    # define the set of 7 segments
    segments = [
        ((0, 0), (w, dH)),  # top
        ((0, 0), (dW, h // 2)),  # top-left
        ((w - dW, 0), (w, h // 2)),  # top-right
        ((0, (h // 2) - dHC), (w, (h // 8) + dHC)),  # center
        ((0, h // 2), (dW, h)),  # bottom-left
        ((w - dW, h // 2), (w, h)),  # bottom-right
        ((0, h - dH), (w, h))  # bottom
    ]

    on = [0] * len(segments)
    # loop over the segments
    for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
        # extract the segment ROI, count the total number of
        # thresholded pixels in the segment, and then compute
        # the area of the segment
        segROI = roi[yA:yB, xA:xB]

        # Debugging visualization
        cv2.rectangle(roi, (xA, yA), (xA + xB, yA + yB), (255, 255, 255), 5)
        cv2.imshow("", roi)
        cv2.waitKey(0)

        total = cv2.countNonZero(segROI)
        area = (xB - xA) * (yB - yA)
        print((total, float(area)))
        # if the total number of non-zero pixels is greater than
        # 50% of the area, mark the segment as "on"
        if abs(total / float(area)) > 0.2:
            on[i] = 1
        # print(on)
    # lookup the digit and draw it on the image
    try:
        digit = imageAnalysis.DIGITS_LOOKUP[tuple(on)]
    except Exception as e:
        # print(e)
        digit = "?"
    print((on, digit))
    # cv2.imshow("", image)
    # cv2.waitKey(0)
    digits.append(digit)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.putText(image, str(digit), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    return digits


def recognize_digit2(image, digit_contour):
    (x, y, w, h) = cv2.boundingRect(digit_contour)
    roi = image[y:y + h, x:x + w]
    resized_roi = cv2.resize(roi, (28, 28))
    resized_roi = resized_roi.reshape(1, 28 * 28)
    digit = model.predict(resized_roi)
    print(numpy.argmax(digit))
    cv2.imshow("test", roi)
    cv2.waitKey(0)
    return digit


def main():
    cap = cv2.VideoCapture('Videos/01.mp4')

    _, firstFrame = cap.read()

    frame, _, _ = preprocess_image(firstFrame, is_first_frame=True)

    while (cap.isOpened()):
        ret, frame = cap.read()

        try:
            preprocessed_image, pre_gray, pre_thresh = preprocess_image(frame)
            # blurred = cv2.GaussianBlur(preprocessed_image, (7, 7), 0)
            edged = imageAnalysis.run_edge_detection(preprocessed_image, 0)
            _, digit_contours = find_digits(edged)

            # gray = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY)

            for cont in digit_contours:
                digit = recognize_digit(pre_gray, cont)
                # print(digit)

            cv2.imshow('frame', preprocessed_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(e)
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

# thresh=imageAnalysis.run_threshold(pre_gray, 40, 240, 1)
