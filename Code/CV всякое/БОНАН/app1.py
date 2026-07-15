import cv2

image = cv2.imread('PCB6.png', cv2.IMREAD_GRAYSCALE)

blurred = cv2.GaussianBlur(image, (5, 5), 0)


binary_pcb = cv2.adaptiveThreshold(
    blurred, 255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY_INV, 31, 5
)
cv2.imwrite('saved_image.png', binary_pcb) 
