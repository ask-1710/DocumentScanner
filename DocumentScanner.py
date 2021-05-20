import cv2
import numpy as np

widthImg = 640
heightImg = 480

cap = cv2.VideoCapture(0)  # instead of path,0 will take default webcam
# defining parameters
cap.set(3, widthImg)  # width id 3->640
cap.set(4, heightImg)  # heigth -id no 4->480
cap.set(10, 150)  # brightness(id-10) ->100


def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    kernel = np.ones((5, 5))
    imgDilation = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThreshold = cv2.erode(imgDilation, kernel, iterations=1)

    return imgThreshold


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
                cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
    return biggest


def reorder(myPoints):
    pass


def getWarp(img, biggest):
    pts1 = np.array(biggest, np.float32)
    pts2 = np.array([[0, heightImg], [0, 0], [widthImg, 0], [widthImg, heightImg]], np.float32)
    # pts1 = np.float32(biggest)
    # pts2 = np.float32([[0, heightImg], [0, 0], [widthImg, 0], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    return imgOutput


while True:
    success, img = cap.read()
    cv2.imshow("Original Image", img)
    cv2.resize(img, (widthImg, heightImg))
    imgContour = img.copy()
    imgThres = preProcessing(img)
    biggest = getContours(imgThres)
    print(biggest)
    cv2.imshow("webcam", imgThres)
    imgWarped = getWarp(img, biggest)
    cv2.imshow("webcam", imgWarped)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
