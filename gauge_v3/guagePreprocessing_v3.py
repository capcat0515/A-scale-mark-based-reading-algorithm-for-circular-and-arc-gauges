import cv2
import numpy as np

def preprocessing_passRW(img):
	imgbin = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,9)
	imgbin=cv2.bitwise_not(imgbin)
	filter_3by3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
	imgbin = cv2.erode(imgbin,filter_3by3,iterations =1)
	imgbin = cv2.dilate(imgbin,filter_3by3,iterations = 1)
	return imgbin


def main():
	pass

if __name__ == "__main__" :
	main()
