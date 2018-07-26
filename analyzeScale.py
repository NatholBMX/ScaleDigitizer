# import the necessary packages
import cv2
import numpy
import urllib.request, urllib.error, urllib.parse
from utils import imageAnalysis

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
        scale_roi = (centerX - radius, centerY - radius, centerX + radius, centerY + radius)
        print(scale_roi)

        cropped_image = rotated_image[scale_roi[0]:scale_roi[0] + scale_roi[2], scale_roi[1]:scale_roi[1] + scale_roi[3]]
        blurred = cv2.GaussianBlur(cropped_image, (7, 7), 0)
        edged = imageAnalysis.run_edge_detection(blurred, 0)
        cv2.imshow("", cropped_image)
        _, x1, x2, y1, y2=imageAnalysis.detect_horizontal_lines(edged)
        scale_roi=(centerX - radius, y1, centerX + radius, y2)
        print(scale_roi)

    processed_image = rotated_image[scale_roi[0]:scale_roi[0] + scale_roi[2], scale_roi[1]:scale_roi[1] + scale_roi[3]]

    return processed_image


def main():
    cap = cv2.VideoCapture('Videos/02.mp4')

    _, firstFrame = cap.read()

    frame = preprocess_image(firstFrame, is_first_frame=True)

    while (cap.isOpened()):
        ret, frame = cap.read()

        try:
            preprocessed_image = preprocess_image(frame)
            #blurred = cv2.GaussianBlur(preprocessed_image, (7, 7), 0)
            #edged=imageAnalysis.run_edge_detection(blurred, 0)

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