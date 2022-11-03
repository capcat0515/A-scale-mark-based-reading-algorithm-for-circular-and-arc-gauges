import cv2
import numpy as np
import math
from collections import Counter
from sklearn.cluster import KMeans
import os


def scaleToCenter(scales, center, C4):
	scalesCenterdistance=[]
	shortAver=0
	longAver=0
	scalesMToC_L=[]
	scalesBox=[]
	for i in range(len(scales)):
		rect = cv2.minAreaRect(scales[i])
		box = cv2.boxPoints(rect)
		box=np.int0(box)
		boxX=0
		boxY=0
		for j in range(len(box)):
			boxX+=box[j][0]
			boxY+=box[j][1]
		boxX=boxX/4
		boxY=boxY/4

		dm=math.sqrt( (boxX- center[0])*(boxX- center[0])+(boxY- center[1])*(boxY- center[1]))
		scalesMToC_L.append(dm)
		scalesBox.append(box)


	for i in range(len(scalesBox)):
		box=scalesBox[i]
		##print(box)
		wlen=math.sqrt((box[0][0]-box[1][0])*(box[0][0]-box[1][0])+(box[0][1]-box[1][1])*(box[0][1]-box[1][1]))
		hlen=math.sqrt((box[1][0]-box[2][0])*(box[1][0]-box[2][0])+(box[1][1]-box[2][1])*(box[1][1]-box[2][1]))
		if wlen < hlen:

			w1= (int( (box[0][0]+box[1][0])/2 ), int( (box[0][1]+box[1][1])/2 ))
			w2= (int( (box[2][0]+box[3][0])/2 ), int( (box[2][1]+box[3][1])/2 ))
		else:
			w1= (int( (box[1][0]+box[2][0])/2 ), int( (box[1][1]+box[2][1])/2 ))
			w2= (int( (box[0][0]+box[3][0])/2 ), int( (box[0][1]+box[3][1])/2 ))

		d1=math.sqrt( (w1[0]- center[0])*(w1[0]- center[0])+(w1[1]- center[1])*(w1[1]- center[1]))
		d2=math.sqrt( (w2[0]- center[0])*(w2[0]- center[0])+(w2[1]- center[1])*(w2[1]- center[1]))
		if d1 < d2:
			shortAver+=d1
			longAver+=d2
		else:
			longAver+=d1
			shortAver+=d2


		x=0
		y=0
		for j in range(len(scales[i])):
			x+=scales[i][j][0][0]
			y+=scales[i][j][0][1]
		x=x/len(scales[i])
		y=y/len(scales[i])
		SCD=math.sqrt((x-center[0])*(x-center[0])+(y-center[1])*(y-center[1]))
		scalesCenterdistance.append(SCD)
	shortAver=int(shortAver/len(scales))
	longAver=int(longAver/len(scales))


	return int(sum(scalesCenterdistance)/len(scales)), shortAver, longAver, C4


def scaleMarks(img, shortAver, longAver, center):
	scalemarkMask=np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
	diff_SL=longAver-shortAver
	cv2.circle(scalemarkMask, center, longAver, (255), -1)
	cv2.circle(scalemarkMask, center, (shortAver-diff_SL), (0), -1)
	img1=cv2.bitwise_and(img,img,mask=scalemarkMask)
	return img1


def main():
	Pass

if __name__ == "__main__" :
	main()