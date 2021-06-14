import argparse
import cv2 as cv
import imutils
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths

truePos = 0
falseNeg = 0

trueNeg = 0
falsePos = 0

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--Fire", required=True,
                help="the required path for finding image")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["Fire"]))

for imagePath in imagePaths:
    # load the image and resize it to (1) reduce detection time
    img = cv.imread(imagePath)
    orig = img.copy()

#bound defined for detecting fire colors
    lower_bound = np.array([11, 55, 180])

    upper_bound = np.array([90, 255, 255])
#resize input image
    img = cv.resize(img, (1280, 720))
    img = cv.flip(img, 1)

#smoothing technique apply
    frame_smooth = cv.GaussianBlur(img, (7, 7), 0)

#masking technique apply
    mask = np.zeros_like(img)

    mask[0:720, 0:1280] = [255, 255, 255]

#ROI apply
    img_roi = cv.bitwise_and(frame_smooth, mask)

    frame_hsv = cv.cvtColor(img_roi, cv.COLOR_BGR2HSV)

#convert to binary image
    image_binary = cv.inRange(frame_hsv, lower_bound, upper_bound)

    check_if_fire_detected = cv.countNonZero(image_binary)

    if int(check_if_fire_detected) < 16500:
        trueNeg = trueNeg + 1
        cv.putText(img, "No Fire!", (200, 90), cv.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 2)
    else:
        truePos = truePos + 1
        cv.putText(img, "Fire!", (200, 90), cv.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 2)

    cv.imshow("Fire Detection", img)

    print("True Pos")
    print(truePos)

    print("True Neg")
    print(trueNeg)

    rc = truePos / (truePos + trueNeg)
    print("Recall ")
    print(rc)

    cv.waitKey(0)
