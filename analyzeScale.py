# import the necessary packages
import cv2
import numpy
import urllib.request, urllib.error, urllib.parse
from utils import imageAnalysis
import imutils

USE_WEBCAM = False

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
    blurred = cv2.GaussianBlur(cropped_image, (7, 7), 0)
    kernel = numpy.ones((7, 7), numpy.uint8)
    processed_image = cv2.erode(blurred, kernel, iterations=1)

    return processed_image

def find_digits(image):
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL,
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
        x, y, w, h = cv2.boundingRect(c)
        if h<image.shape[0]/5:
            continue
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
    return image


def main():
    cap = cv2.VideoCapture('Videos/02.mp4')

    _, firstFrame = cap.read()

    frame = preprocess_image(firstFrame, is_first_frame=True)

    while (cap.isOpened()):
        ret, frame = cap.read()

        try:
            preprocessed_image = preprocess_image(frame)
            # blurred = cv2.GaussianBlur(preprocessed_image, (7, 7), 0)
            edged = imageAnalysis.run_edge_detection(preprocessed_image, 0)
            find_digits(edged)

            cv2.imshow('frame', edged)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(e)
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
