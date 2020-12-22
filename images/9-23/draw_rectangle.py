import cv2

img = cv2.imread('./3-1.jpg')

newimg = cv2.rectangle(img, (400, 214), (400 + 1528, 214 + 1010), (0, 0, 255), 5)

cv2.imwrite('3-1-rectangel.jpg', newimg)


