import cv2
import imutils
import argparse
import numpy as np
from operator import itemgetter
import imutils.perspective

def get_biggest_difference(lines):
    #there's probably an OpenCV function for this
    xs = [i[0] for i in lines]
    biggest = 0
    biggestid = 0
    for i in range(1, len(xs)):
        dif = xs[i] - xs[i - 1]
        if dif > biggest:
            biggest = dif
            biggestid = i

    return biggestid - 1, biggestid

def contours_to_points(contour):
    #this is NOT how you are supposed to do this
    pts = str(contour).split("\n")
    pts = sorted([[int(i) for i in pt.replace("[", "").replace("]", "").split(" ") if i != ""] for pt in pts if pt != ""], key = itemgetter(1))
    return [[pts[0][0], pts[0][1] + bbar], [pts[-1][0], pts[-1][1] + bbar]]

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", help = "Path to the video file. ")
args = vars(parser.parse_args())
camera = cv2.VideoCapture(args["video"])

while True:
    cap, img = camera.read()
    if not cap:
        break

    #blur and convert to grayscale
    gray = cv2.GaussianBlur(img, (11, 11), 0)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    #canny edge detection
    canny = cv2.Canny(gray, 3, 25)

    #hough lines
    lines = cv2.HoughLines(canny,1,np.pi/180,150)
    linecoords = []
    for line in lines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            linecoords.append([x1, y1, x2, y2])
    
    #find the biggest space between lines
    lines = sorted(linecoords, key = itemgetter(0))
    biggest_difference = get_biggest_difference(lines)

    y, x, _ = img.shape

    #   p1------p2
    #   |        |
    #   |        |
    #   p3------p4

    p1 = (lines[biggest_difference[0]][0], lines[biggest_difference[0]][1])
    p3 = (lines[biggest_difference[0]][2], lines[biggest_difference[0]][3])
    p2 = (lines[biggest_difference[1]][0], lines[biggest_difference[1]][1])
    p4 = (lines[biggest_difference[1]][2], lines[biggest_difference[1]][3])
    
    orig = img.copy()
    lines = img.copy()
    lines2 = img.copy()
    #draw lines with biggesr space on lines with green lines
    lines = cv2.line(lines, p1, p3, (0, 255, 0), 1)
    lines = cv2.line(lines, p2, p4, (0, 255, 0), 1)
    
    #purely to see what's going on, also draw on a normal image
    lines2 = cv2.line(lines2, p1, p3, (0, 255, 0), 2)
    lines2 = cv2.line(lines2, p2, p4, (0, 255, 0), 2)

    #threshold lines image and find contours
    lines = cv2.cvtColor(lines, cv2.COLOR_BGR2HSV)
    lines = cv2.inRange(lines, (0, 255, 255), (255, 255, 255))
    cnts = cv2.findContours(lines.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    #draw black bars
    bbar = 20
    img = cv2.copyMakeBorder(img, bbar, bbar, 0, 0, cv2.BORDER_CONSTANT)
    orig = cv2.copyMakeBorder(orig, bbar, bbar, 0, 0, cv2.BORDER_CONSTANT)

    #only proceed if two contours are found
    if len(cnts) == 2:
        #convert contours into strings and append them to a list
        cntpts = []
        for c in cnts:
            pts = contours_to_points(c)
            cntpts.append(pts)
    
        #convert list into one big contour
        allcontours = np.array(cntpts).reshape((-1,1,2)).astype(np.int32)

        #draw find bounding box
        rect = cv2.minAreaRect(allcontours)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

        #perspective transform
        persp = imutils.perspective.four_point_transform(orig, box.reshape(4, 2))

        #rotate image
        py, px, _ = persp.shape
        if py > px:
            persp = imutils.rotate_bound(persp, 90)

    cv2.imshow("lines", lines2)
    cv2.imshow("Press 'q' to exit", img)
    try:
        cv2.imshow("persp", persp)
    except NameError:
        pass
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
