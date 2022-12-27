# Import dependencies
import cv2
# Read Images
img = cv2.imread('biaoding/zuo/left_2000.jpg')
# Display Image
if img is None:
    print('Could not read image')
# Draw line on image
imageLine = img.copy()
# Draw the image from point A to B
pointA = (150,200)
pointB = (150,200)
cv2.line(imageLine, pointA, pointB, (255, 255, 0), thickness=3, lineType=cv2.LINE_AA)
cv2.imshow('Image Line', imageLine)
cv2.waitKey(0)
