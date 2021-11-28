import cv2 as cv
#将白底黑字的图片转成黑底白字的图片
src = cv.imread('9.png')
gray = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
ret,bin = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
cv.imwrite('new9.png',bin)
