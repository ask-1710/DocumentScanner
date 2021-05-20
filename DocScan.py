import cv2
import numpy as np

widthImg = 640
heightImg = 480

filename = 'Resources/book2.jpeg'
img = cv2.imread(filename)


def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    kernel = np.ones((5, 5))
    imgDilation = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThreshold = cv2.erode(imgDilation, kernel, iterations=1)

    return imgThreshold


def reorder(myPoints):

    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew


def getContours(img):
    maxArea = 0
    biggest = np.array([])
    contours, heirarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            per1 = cv2.arcLength(cnt, True)
            approx = (cv2.approxPolyDP(cnt, 0.02 * per1, True))  # approx of corners
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
                biggest = reorder(biggest)
                cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
    return biggest


def getWarp(img, biggest):
    pts1 = np.float32(biggest)
    pts2 = np.float32([[widthImg, heightImg], [0, heightImg], [widthImg, 0],  [0, 0]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    return imgOutput


cv2.resize(img, (widthImg, heightImg))
imgContour = img.copy()
imgThres = preProcessing(img)
biggest = getContours(imgThres)
imgWarped = getWarp(img, biggest)
cv2.imshow("Scanned Document", imgWarped)
cv2.waitKey(0)

