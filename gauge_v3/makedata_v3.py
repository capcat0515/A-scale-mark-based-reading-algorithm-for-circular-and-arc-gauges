import sys
import cv2
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import pandas as pd
import csv

def ans(t, n):
	AA=[ [1.00,	1.37, 0.47], [0.75, 0.40, -0.42], [0.30, 0.15, 0.00], [5.00, 2.20, 7.00] ]
	BB=[6,4,2.5,50]
	return AA[t][n], BB[t]

def namesort(ccc, t):
	axisD={'x':0, 'y':1}
	ccc1=[]
	for i in range(6):
		ccc1.append([])
	for ci in ccc:
		names=ci[0].split('_')
		#print(int(names[1])*2,int(axisD[names[2]]))
		nums=(int(names[1])*2+int(axisD[names[2]]))%6
		GVans, GVrange=ans(t,int(nums/2))
		if(len(ci)==4 and ci[1]!=None):
			ci.append(round(abs(GVans-ci[1])/GVrange, 4))
			pass
			#ci.append(0.02)
		else:
			ci.append(-0.1)
			#ci.append(-0.1)
			#ci.append(-0.1)
			#ci.append(-0.1)
		ccc1[nums].append(ci)
		#print(nums)
	tc=[]
	tl=[[],[]]
	#print(ccc1)
	if len(names) == 4:
		for i in range(6):
			for j in range(-45, 46, 3):
				
				for kk in ccc1[i]:
					names=kk[0].split('_')
					if(180<int(names[-1])<360):
						nn=int(names[-1])-360
					else:
						nn=int(names[-1])
					#print(nn)
					if(nn == j):
						tc.append(kk)
						break
	else:
		for i in range(6):
			for j in range(-45, 46, 3):
				n=0
				for kk in ccc1[i]:
					names=kk[0].split('_')
					if(180<int(names[-3])<360):
						nn=int(names[-3])-360
					else:
						nn=int(names[-3])
					if(names[-3]==names[-2] and nn==j):
						tl[0].append(kk)
						n+=1
					if(int(names[-3])+int(names[-2])==360 and nn==j):
						tl[1].append(kk)
						n+=1
					if(n==2):
						break
		for i in range(len(tl)):
			for j in tl[i]:
				tc.append(j)
	
	return tc


def d1d2(center, d1, d2, img):
	imgmask = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
	cv2.circle(imgmask, center, int(d1), (255), -1)
	cv2.circle(imgmask, center, int(d2), (0), -1)
	imgROI=cv2.bitwise_and(img,img,mask=imgmask)
	return imgROI
def d1d2_draw(center, d1, d2, img):
	#imgmask = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
	cv2.circle(img, center, int(d1), (128), 2)
	cv2.circle(img, center, int(d2), (128), 2)
	#imgROI=cv2.bitwise_and(img,img,mask=imgmask)
	return img

def SMratio(center, d1, d2, img, toGC_cc_num):
	imgc=img.copy()
	imgmask1 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
	cv2.circle(imgmask1, center, int(d1), (255), -1)
	cv2.circle(imgmask1, center, int(d2), (0), -1)
	imgmask2 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
	cv2.circle(imgmask2, center, int(d1), (255), -1)

	imgROI1=cv2.bitwise_and(img,img,mask=imgmask1)
	imgROIdots=int(np.sum(imgROI1)/255)
	imgROI2=cv2.bitwise_and(img,img,mask=imgmask2)
	imgdots=int(np.sum(imgROI2)/255)
	dots=3874
	if(toGC_cc_num>0):
		#SMR=round((imgROIdots/imgdots)/toGC_cc_num, 4)
		SMR=round((imgROIdots)/toGC_cc_num, 4)
	else:
		SMR=0
	#SMR=round(imgROIdots/imgdots, 2)
	#SMR=round(imgdots/dots, 2)


	cv2.circle(imgc, center, int(d1), (128), 1)
	cv2.circle(imgc, center, int(d2), (128), 1)
	return SMR, imgc, imgdots


def max_min_angle(infolist):
	
	axisD={'x':0, 'y':1}
	nn=infolist[0][0].split('_')
	angles=[[],[]]
	if len(nn) == 4:
		for i in range(2):
			minA=-1
			maxA=-1
			for info in infolist:
				name=info[0].split('_')
				ax=axisD[name[2]]
				num=int(name[-1])
				if(num>180):
					num=abs(num-360)
				if(ax ==i):
					if(minA==-1 ):
						minA=num
						maxA=num
					if(minA>num):
						minA=num
					if(maxA<num):
						maxA=num
			angles[i].append(minA)
			angles[i].append(maxA)
	#else:

	return angles


def drawMSM(guageImg,topNMeanCenter, MSmatchKeysL, color):
	for i in range(len(MSmatchKeysL)):
		aaa=MSmatchKeysL[i]#-90
		aaa=360-aaa
		xxx=math.sin(math.radians(aaa))
		yyy=math.cos(math.radians(aaa))
		#print(xxx,yyy, aaa)
		cv2.line(guageImg, topNMeanCenter, (int(topNMeanCenter[0]*xxx*1000), int(topNMeanCenter[1]*yyy*1000)), color, 1)
	return guageImg

def computegap(MSmatchKeysL):
	halfgap=0
	for i in range(1,len(MSmatchKeysL)):
		halfgap+=MSmatchKeysL[i]-MSmatchKeysL[i-1]
	halfgap=int((halfgap/(len(MSmatchKeysL)-1))/3)
	return halfgap

def errorGap(MSmatchKeysL, MScalesLL, halfgap):
	Asum=0
	AsumC=0
	MML=np.zeros(len(MScalesLL))
	errorgapLarge=0
	errorgapL=[]
	errorgapP=[]
	DVLL=[]
	gapaver=0
	newMSMAverA=0
	for i in range(1, len(MSmatchKeysL)):
		gapaver=gapaver+(MSmatchKeysL[i]-MSmatchKeysL[i-1])
	gapaver/=len(MSmatchKeysL)-1
	for i in range(len(MSmatchKeysL)):
		tempAngle=MSmatchKeysL[i]
		smallA=halfgap
		for j in range(len(MScalesLL)):
			if(smallA>abs(MScalesLL[j]-tempAngle) and MML[j]==0):
				smallA=abs(MScalesLL[j]-tempAngle)
				#smallAp=round(smallA/gapaver, 4) 
				smallAp=round(smallA/gapaver, 4) 
				MML[j]=1
		if(smallA<halfgap):
			Asum+=smallA
			AsumC+=1
			errorgapL.append(round(smallA, 3))
			errorgapP.append(smallAp)
			DVLL.append(gapaver)
			if(errorgapLarge<smallA):
				errorgapLarge=smallA

	if(AsumC>1):
		errorgap=round(Asum/AsumC, 4)

	else:
		errorgap=-0.1
		errorgapLarge=-3
	return errorgap, errorgapL, errorgapP,DVLL


def diffMSM(MSML):
	dMSM=[]
	sortMSML=sorted(MSML)
	for i in range(1, len(sortMSML)):
		dMSM.append(sortMSML[i]-sortMSML[i-1])
	return dMSM

def newMSM(MSMD, newMSML):
	MSMDkeys=list(MSMD.keys())
	MSMDgap=MSMDkeys[1]-MSMDkeys[0]
	MSMDvalues=list(MSMD.values())
	newMSMD={}
	boolL=np.zeros(len(MSMDkeys))
	gap=int((MSMDkeys[1]-MSMDkeys[0])/3)
	for i in range(len(newMSML)):
		for j in range(len(MSMDkeys)):
			diffA=abs(newMSML[i]-MSMDkeys[j])
			if(diffA<gap and boolL[j]!=1):
				newMSMD[newMSML[i]]=MSMDvalues[j]
				boolL[j]=1

	newMSMDKeys=list(newMSMD.keys())
	newMSMDKeysdiff=[]
	if len(newMSMDKeys)>1:
		#checkNewMSM()
		newMSMgap=newMSMDKeys[1]-newMSMDKeys[0]
		for i in range(len(newMSMDKeys)):
			newMSMDKeysdiff.append(newMSMDKeys[i]-newMSMDKeys[i-1])
	else:
		newMSMgap=-1

	
	return newMSMD, newMSMgap, newMSMDKeysdiff
