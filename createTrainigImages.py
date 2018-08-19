"""
Script for running over all videos and cropping images for training
"""

import os
import cv2
import numpy
from utils import imageAnalysis
import imutils
import analyzeScale

VIDEOS_DIR_PATH= "./videos/"
TRAINING_DIR_PATH="./training/"

def extract_digits(gray_image, digit_positions):
    training_img_list=os.listdir(TRAINING_DIR_PATH)

    if len(training_img_list)==0:
        counter=0
    else:
        counter=int(training_img_list[-1].split(".")[0])

    for digit in digit_positions:
        x0, y0 = digit[0]
        x1, y1 = digit[1]
        roi = gray_image[y0:y1, x0:x1]
        cv2.imwrite(TRAINING_DIR_PATH+"%5d" % counter+".jpg", roi)
        counter+=1



def main():
    if not os.path.isdir(TRAINING_DIR_PATH):
        os.makedirs(TRAINING_DIR_PATH)
    video_list=os.listdir(VIDEOS_DIR_PATH)

    for video in video_list:

        cap = cv2.VideoCapture(VIDEOS_DIR_PATH + video)
        _, firstFrame = cap.read()

        frame, _ = analyzeScale.crop_scale2(firstFrame, True)
        digits = []
        try:
            while (cap.isOpened()):
                ret, frame = cap.read()
                alpha = float(1.6)
                frame = cv2.multiply(frame, numpy.array([alpha]))

                preprocessed_image, pre_gray = analyzeScale.crop_scale2(frame)

                pre_gray = cv2.resize(pre_gray, (0, 0), fx=2, fy=2)
                dst = analyzeScale.preprocess_image(pre_gray)
                digits_positions = analyzeScale.find_digits(dst)
                extract_digits(dst, digits_positions)

                #digits.append(analyzeScale.recognize_digits_line_method(digits_positions, pre_gray, dst))

        except Exception as e:
            print(e)



if __name__ == '__main__':
    main()
