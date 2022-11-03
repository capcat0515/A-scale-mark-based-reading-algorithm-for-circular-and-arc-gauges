import cv2
import numpy as np
import math
import os

def CCsParameter(cc):
	rect = cv2.minAreaRect(cc)
	box = cv2.boxPoints(rect)
	box=np.int0(box)
	wlen=math.sqrt((box[0][0]-box[1][0])*(box[0][0]-box[1][0])+(box[0][1]-box[1][1])*(box[0][1]-box[1][1]))
	hlen=math.sqrt((box[1][0]-box[2][0])*(box[1][0]-box[2][0])+(box[1][1]-box[2][1])*(box[1][1]-box[2][1]))
	breakbool=0
	if wlen > hlen:
		wlen, hlen = hlen, wlen
	if hlen != 0:
		wh=wlen/hlen*1.0
	else:
		wh=0
		breakbool=1
	M = cv2.moments(cc)
	if M['m00'] != 0:
		cx = int(M['m10']/M['m00'])
		cy = int(M['m01']/M['m00'])
	else:
		cx=cy=0
		breakbool=1
	return breakbool, rect, box, wlen, hlen, wh, cx, cy

def ccPCA(cc):
	sz = len(cc)
	data_pts = np.empty((sz, 2), dtype=np.float64)
	for ii in range(data_pts.shape[0]):
		data_pts[ii,0] = cc[ii,0,0]
		data_pts[ii,1] = cc[ii,0,1]
	mean = np.empty((0))
	mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
	return mean, eigenvectors, eigenvalues

def ccDegree(GCenter, mean, eigenvectors):
	meanV=(GCenter[0]-mean[0][0], GCenter[1]-mean[0][1])
	meanLen=math.sqrt((GCenter[0]-mean[0][0])*(GCenter[0]-mean[0][0])+(GCenter[1]-mean[0][1])*(GCenter[1]-mean[0][1]))
	angles=np.degrees(np.arctan2(eigenvectors[0][1], eigenvectors[0][0]))
	angle2=np.degrees(np.arctan2(meanV[1], meanV[0]))
	if(angles<0):
		angles+=180
	if(angle2<0):
		angle2+=180

	anglediff=abs(int(angles)-int(angle2))
	#print(int(angles),int(angle2))
	return anglediff, meanLen

def pointerAngle(cc, GCenter):
	disL=[]
	for i in range(len(cc)):
		x=cc[i][0][0]
		y=cc[i][0][1]
		disL.append((GCenter[0]-x)*(GCenter[0]-x)+(GCenter[1]-y)*(GCenter[1]-y))
	tmp = max(disL)
	index = disL.index(tmp)
	x=cc[index][0][0]
	y=cc[index][0][1]
	pp=(int(x), int(y))
	middleDegrees= math.degrees(math.atan2((GCenter[1]-y),(GCenter[0]-x))) #20220206 modify
	return middleDegrees , pp

def findGaugeNiddle(binImg, cnt, GCenter, averStoC_D, clearImg1, clearImg, shortAver, longAver):
	rectD=[]

	picA=binImg.shape[0]*binImg.shape[1]
	diffAver=longAver-shortAver
	diffAverTwice=diffAver*3 #d
	scalesRegionL=shortAver-diffAver #d
	filter5= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
	
	#the region of G with radius d_1 to be G_ROI
	scalemarkMask=np.zeros((binImg.shape[0], binImg.shape[1], 1), np.uint8)
	cv2.circle(scalemarkMask, GCenter, (longAver), (255), -1)
	binImgCopy=binImg.copy()
	binImgCopy=cv2.bitwise_and(binImgCopy,binImgCopy,mask=scalemarkMask)
	clearImgROI=cv2.bitwise_and(clearImg,clearImg,mask=scalemarkMask)
	binImgROI=binImgCopy.copy()
	
	#find CC
	kernel = np.ones((5,5), np.uint8)
	cnt,hierarchy = cv2.findContours(binImgCopy, 1, 2)
	maybeNiddleRegion = np.zeros((binImg.shape[0], binImg.shape[1], 1), np.uint8)
	anglesOff=10
	ccCondiL=[]
	meanL=[]
	for i in range(len(cnt)):
		breakbool, rect, box, wlen, hlen, wh, cx, cy=CCsParameter(cnt[i])
		tempA=cv2.contourArea(cnt[i])
		whArea=wlen*hlen
		CompactnessN=0
		if(whArea>0):
			CompactnessN=round(tempA/whArea, 2)
		mean, eigenvectors, eigenvalues=ccPCA(cnt[i])

		meanL.append(mean)

		largelen=-1
		furcc=None
		for j in range(len(cnt[i])):
			LLen=math.sqrt((GCenter[0]-cnt[i][j][0][0])*(GCenter[0]-cnt[i][j][0][0])+(GCenter[1]-cnt[i][j][0][1])*(GCenter[1]-cnt[i][j][0][1]))
			if(LLen>largelen):
				largelen=LLen
				furcc=cnt[i][j]
		anglediff, meanLen=ccDegree(GCenter, furcc, eigenvectors)
		anglediff2, meanLen=ccDegree(mean[0], furcc, eigenvectors)
	
		if(wh<0.33 and (anglediff<anglesOff or anglediff>180-anglesOff ) and (anglediff2<10 or anglediff2>170)):
			
			ccCondiL.append(i)
	pointerArea=-1
	pointerAreaID=-1
	pointerwh=-1
	
	for i in range(len(ccCondiL)):
		
		tempA=cv2.contourArea(cnt[ccCondiL[i]])
		
		
		
		cv2.fillPoly(maybeNiddleRegion,[cnt[ccCondiL[i]]],(255))
		
		if(pointerArea<0 ):
			pointerArea=tempA
			pointerAreaID=ccCondiL[i]
		if(tempA>pointerArea ): 
			pointerArea=tempA
			pointerAreaID=ccCondiL[i]

	if(pointerAreaID>=0):
		cv2.fillPoly(maybeNiddleRegion,[cnt[pointerAreaID]],(255))
		cv2.circle(maybeNiddleRegion, (int(meanL[pointerAreaID][0][0]), int(meanL[pointerAreaID][0][1])), 5, (128), -1)
		offsetMiddleAngle1, pp=pointerAngle(cnt[pointerAreaID], GCenter)
		cv2.line(maybeNiddleRegion, GCenter, pp, (128), 2)
	if(pointerAreaID==-1):
		offsetMiddleAngle1=None
	return offsetMiddleAngle1, maybeNiddleRegion


def main():
	pass
	
if __name__ == "__main__" :
	main()
