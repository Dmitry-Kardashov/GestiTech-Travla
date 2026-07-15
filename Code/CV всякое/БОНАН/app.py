import cv2
image = cv2.imread('PCB.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray_image, 130, 255, cv2.THRESH_BINARY)

thresh_gaussian = cv2.adaptiveThreshold(
    gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY, 11, 2
)

cv2.imwrite('saved_image.png', thresh_gaussian) 