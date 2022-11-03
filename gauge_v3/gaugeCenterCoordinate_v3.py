import cv2
import numpy as np
import math
from collections import Counter
from sklearn.cluster import KMeans
import time

def getOrientation(pts):
	data_pts = np.empty((len(pts), 2), dtype=np.float64)
	for i in range(data_pts.shape[0]):
		data_pts[i,0] = pts[i,0,0]
		data_pts[i,1] = pts[i,0,1]
	# Perform PCA analysis
	mean = np.empty((0))
	mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
	# Store the center of the object
	cntr = (int(mean[0,0]), int(mean[0,1]))
	p1 = (int(cntr[0] + eigenvectors[0,0]*100), int(cntr[1] +eigenvectors[0,1]*100))
	Lines=[cntr[0], cntr[1], int(cntr[0] + eigenvectors[0,0]*100), int(cntr[1] +eigenvectors[0,1]*100)]

	return Lines, mean, eigenvectors

def cross_point(line1,line2):
	Linelen=-1
	x1,y1,x2,y2=line1 
	x3,y3,x4,y4=line2
	if((x2-x1)!=0 and (x4-x3)!=0):
		k1=(y2-y1)*1.0/(x2-x1)
		b1=y1*1.0-x1*k1*1.0
		k2=(y4-y3)*1.0/(x4-x3)
		b2=y3*1.0-x3*k2*1.0
		if((k1-k2)!=0):
			x=(b2-b1)*1.0/(k1-k2)
			y=k1*x*1.0+b1*1.0
			Linelen = (int(x), int(y))
	return Linelen

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
	return anglediff

def ccToGC(GC, Ms, EVs, CC, img, d1, d2): #20220703
	num=0
	numd1d2=0
	diffA=20
	for i in range(len(Ms)):
		angle=ccDegree(GC, Ms[i], EVs[i])
		if(angle<diffA or diffA>180-diffA):
			bool1=0
			#print(126)
			for k in range(len(CC[i])):
				#print(CC[i][k][0])
				dis=math.sqrt((GC[0]-CC[i][k][0][0])*(GC[0]-CC[i][k][0][0])+(GC[1]-CC[i][k][0][1])*(GC[1]-CC[i][k][0][1]))
				if(dis<d1):
					bool1=1
					break
			if(bool1==1):
				cv2.fillPoly(img,[CC[i]],(255,255,255))
				num+=1
			dis1=math.sqrt((GC[0]-Ms[i][0][0])*(GC[0]-Ms[i][0][0])+(GC[1]-Ms[i][0][1])*(GC[1]-Ms[i][0][1]))
			if(d1>dis1>d2):
				numd1d2+=1

	return num, img, numd1d2

def GC_CC(img, GC, d1, d2):
	newSM = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
	means=[]
	EVs=[]
	contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	for i, cc in enumerate(contours):
		Lines, mean, eigenvectors=getOrientation(cc)
		means.append(mean)
		EVs.append(eigenvectors)
	toGC_cc_num, tempimg, numd1d2=ccToGC(GC, means, EVs, contours, newSM, d1, d2)
	return toGC_cc_num, tempimg, numd1d2

def gaugeCenterPoint(outputscalemarks,ccList,clearImg1, N):
	contours, _ = cv2.findContours(outputscalemarks, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	PCALines=[]
	centerPoints=[]
	
	for i, cc in enumerate(contours):
		Lines, mean, eigenvectors=getOrientation(cc)
		PCALines.append(Lines)

	for i in range(len(PCALines)-1):
		for j in range(i+1, len(PCALines)):
			CP=cross_point(PCALines[i], PCALines[j])
			if(CP != -1):
				centerPoints.append(CP)
	
	mostPointsV1 = Counter(centerPoints)
	mostPointsV2 = mostPointsV1.most_common(N)
	gaugeCenterX=0
	gaugeCenterY=0
	for i in range(N):	
		gaugeCenterX+=mostPointsV2[i][0][0]
		gaugeCenterY+=mostPointsV2[i][0][1]
	gaugeCenterX=int(gaugeCenterX/N)
	gaugeCenterY=int(gaugeCenterY/N)
	cv2.circle(clearImg1, (gaugeCenterX, gaugeCenterY), 1, (0,255,0), 10, cv2.LINE_AA)
	
	return clearImg1, (gaugeCenterX, gaugeCenterY)

def main():
	pass

if __name__ == "__main__" :
	main()