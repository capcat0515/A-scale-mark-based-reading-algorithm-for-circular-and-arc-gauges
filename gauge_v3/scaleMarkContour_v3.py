import cv2
import numpy as np
import math
from collections import Counter
import matplotlib.pyplot as plt
from math import atan2, cos, sin, sqrt, pi, degrees
from mpl_toolkits.mplot3d import axes3d

def ccParameter(cc):
	#Area
	area=cv2.contourArea(cc)

	#
	epsilon = 0.01*cv2.arcLength(cc,True) 
	approx = cv2.approxPolyDP(cc,epsilon,True)

	#whRatio and compactness
	whRatio=0
	compactness=0
	box = np.int0(cv2.boxPoints(cv2.minAreaRect(cc)))
	wlen=math.sqrt((box[0][0]-box[1][0])*(box[0][0]-box[1][0])+(box[0][1]-box[1][1])*(box[0][1]-box[1][1]))
	hlen=math.sqrt((box[1][0]-box[2][0])*(box[1][0]-box[2][0])+(box[1][1]-box[2][1])*(box[1][1]-box[2][1]))
	if wlen > hlen:
		wlen, hlen = hlen, wlen
	if hlen!=0 and wlen!=0:
		whRatio=round(wlen/(hlen*1.0), 3)
		compactness=round(area/(wlen*hlen*1.0), 3) #compactness
	return area, whRatio, compactness, wlen, hlen, approx




def findScaleMarksContour(binImg):
	outputColor4 = np.zeros((binImg.shape[0], binImg.shape[1], 1), np.uint8)
	contours,hierarchy = cv2.findContours(binImg, 1, 2)
	contoursList=[]
	hisList=[]
	wlenL=[]
	hlenL=[]
	areaP=int(binImg.shape[0]* binImg.shape[1]*0.0001)
	areaSum=0
	for i in range(len(contours)):
		area=cv2.contourArea(contours[i])
		areaSum+=area
	areaSum=areaSum*0.0001
	if areaSum < 10:
		areaSum=10

	for i in range(len(contours)):
		area, whRatio, compactness, wlen, hlen,approx=ccParameter(contours[i])
		#print(area, areaP, areaSum)
		if  (compactness >0.5 or len(approx)==8) and ( area> areaSum): #and area >100: area > areaP or
			contoursList.append(contours[i])
			hisList.append(round(whRatio, 1))
			wlenL.append(round(wlen, 1))
			hlenL.append(round(hlen, 1))
	
	#find the highest whRatio values
	mostV1 = Counter(hisList)
	mostV2 = mostV1.most_common(1)
	
	#find the highest whRatio contours
	boolP= np.zeros(len(hisList), np.uint8)
	ROIArea=[]
	ROIAreaN=[] #相同Area個數
	for i in range(len(hisList)):
		if mostV2[0][0] == hisList[i]:
			RBB=wlenL[i]*hlenL[i]
			temp=[]
			tempN=0
			for j in range(len(hisList)):
				if mostV2[0][0] == hisList[j]:
					RBBT=wlenL[j]*hlenL[j]
					if RBB*0.8 < RBBT < RBB*1.2 and boolP[j]==0:
						temp.append([RBBT, contoursList[j]])
						boolP[j]=1
						tempN+=1
			ROIArea.append(temp)
			ROIAreaN.append(tempN)
	scaleMarksList=[]
	shortScaleAreaL=[] #20220127 add
	for j in range(2):
		biggest=-1
		BID=-1
		for k in range(len(ROIArea)): 
			if biggest < ROIAreaN[k]:
				biggest=ROIAreaN[k]
				BID=k
		if BID >= 0:
			ROIAreaN[BID]= -1
			for k in range(len(ROIArea[BID])):
				outputColor4=cv2.drawContours(outputColor4,[ROIArea[BID][k][1]],0,(int(255)),-1)
				shortScaleAreaL.append(round(cv2.contourArea(ROIArea[BID][k][1]),0)) #20220127 add
				scaleMarksList.append(ROIArea[BID][k][1])

	CL=[scaleMarksList, contours]
	return outputColor4, CL, shortScaleAreaL

def countScalsColorValue(grayGuageImg, onlyScalemarkImg, topNMeanCenter, longAver, shortAver):
	diffSL=longAver-shortAver
	onlyScalemarkImgmask=np.zeros((grayGuageImg.shape[0], grayGuageImg.shape[1], 1), np.uint8)
	cv2.circle(onlyScalemarkImgmask, topNMeanCenter, int(longAver+diffSL), (255), -1)
	cv2.circle(onlyScalemarkImgmask, topNMeanCenter, int(shortAver-diffSL), (0), -1)
	onlyScalemarkImg=cv2.bitwise_and(onlyScalemarkImg,onlyScalemarkImg,mask=onlyScalemarkImgmask)

	filter_3by3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
	img_1D=onlyScalemarkImg.ravel()
	count=np.sum(img_1D)/255
	ggI=cv2.bitwise_and(grayGuageImg,grayGuageImg, mask=onlyScalemarkImg)
	img_1D=ggI.ravel()
	if(count>0):
		aver=int(np.sum(img_1D)/count)
	else:
		aver=0
	return aver

def getOrientation(pts): #img
    #print(1)
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    

    angle = degrees(atan2(eigenvectors[0,1], eigenvectors[0,0])) # orientation in radians of degrees
    if len(eigenvalues)==2:
    	return angle, cntr,  eigenvalues[0,0], eigenvalues[1,0]
    else:
    	return angle, cntr,  eigenvalues[0,0], 0

def mergeList(a,b):
	a.extend(b)
	for i in range(len(a)):
		a[i]%=360
	a=sorted(list(set(a)))
	return a

def checkOtherMainscales(maybeMainscalesAngleLSortCopy1, MainscalesAngleLCopy, angleTH, MainscalesAngleL,maybeMainscalesAngleLSort):

	addL=[]
	for i in range(len(MainscalesAngleLCopy)-1):
		compareT=MainscalesAngleLCopy[i]
		#print(i, MainscalesAngleLCopy[i])
		tL=[]
		if(MainscalesAngleLCopy[i+1]-compareT>angleTH+10):
			for j in range(len(maybeMainscalesAngleLSortCopy1)):
				if(compareT<maybeMainscalesAngleLSortCopy1[j]<MainscalesAngleLCopy[i+1]):
					tL.append(maybeMainscalesAngleLSortCopy1[j])
			for j in range(len(tL)):
				
				if(angleTH-10>tL[j]-compareT>angleTH+10):
					addL.append(tL[j])
					compareT=tL[j]
	###print('MainscalesAngleL', MainscalesAngleL)
	MainscalesAngleL=mergeList(MainscalesAngleL, addL)
	###print('MainscalesAngleL', MainscalesAngleL)

	tL=[]
	addL=[]
	tlen=MainscalesAngleL[0]+(360-MainscalesAngleL[-1])
	if(tlen>angleTH+10):
		for i in range(len(maybeMainscalesAngleLSort)):
			if(maybeMainscalesAngleLSort[i]<MainscalesAngleL[0] or MainscalesAngleL[-1]<maybeMainscalesAngleLSort[i]<360):
				if(maybeMainscalesAngleLSort[i]<MainscalesAngleL[0]):
					tn=MainscalesAngleL[0]-maybeMainscalesAngleLSort[i]
				else:
					tn=360-maybeMainscalesAngleLSort[i]+MainscalesAngleL[0]
				tL.append(tn)
		t=0
		for i in range(len(tL)):
			if(angleTH-10<tL[i]-t<angleTH+10):
				addL.append(tL[i])
				t=tL[i]

	MainscalesAngleL=mergeList(MainscalesAngleL, addL)
	###print('MainscalesAngleL', MainscalesAngleL)
	


def subRegion(AdTHimg,shortAver, longAver, center, shortScaleAreaL, avershortcolor, onlyScalemarkImg, colorimg):
	pics=[]
	kernel = np.ones((3,3), np.uint8)
	shortScalesMedian=np.median(shortScaleAreaL)
	shortScalesMean=np.mean(shortScaleAreaL)

	scalemarkMask=np.zeros((AdTHimg.shape[0], AdTHimg.shape[1], 1), np.uint8)
	scalemarkMaskWhite=np.zeros((AdTHimg.shape[0], AdTHimg.shape[1], 1), np.uint8)
	scalemarkMaskWhite.fill(255)
	scalemarkClear=np.zeros((AdTHimg.shape[0], AdTHimg.shape[1], 1), np.uint8)
	scalemarks=np.zeros((AdTHimg.shape[0], AdTHimg.shape[1], 1), np.uint8)
	scalemarks_t=np.zeros((AdTHimg.shape[0], AdTHimg.shape[1], 1), np.uint8)
	Scales=np.zeros((AdTHimg.shape[0], AdTHimg.shape[1], 1), np.uint8)
	diff_SL=longAver-shortAver
	outB=longAver

	inB=shortAver-diff_SL*2
	cv2.circle(scalemarkMask, center, (outB), (255), -1)
	cv2.circle(scalemarkMask, center, (inB), (0), -1)
	
	subAdTHimg=cv2.bitwise_and(AdTHimg,AdTHimg,mask=scalemarkMask)
	#subAdTHimg=cv2.dilate(subAdTHimg, kernel, iterations = 1)
	#subAdTHimg=cv2.erode(subAdTHimg, kernel, iterations = 1)
	scalemarkMaskWhite=cv2.bitwise_and(scalemarkMaskWhite,scalemarkMaskWhite,mask=scalemarkMask)
	scalemarkMaskWhite=cv2.bitwise_not(scalemarkMaskWhite)
	subAdTHimg=cv2.add(subAdTHimg, scalemarkMaskWhite)


	
	scalemarkInsideColor=subAdTHimg.copy()
	pics.append(scalemarkInsideColor)#20220307

	scalemarkMask2=np.zeros((AdTHimg.shape[0], AdTHimg.shape[1], 1), np.uint8)
	cv2.circle(scalemarkMask2, center, (shortAver), (255), -1)
	scalemarkInside=cv2.bitwise_and(colorimg,colorimg,mask=scalemarkMask2)
	pics.append(scalemarkInside)#20220307

	oiArea=int((outB*outB-inB*inB)*3.14)
	sumColorV=0
	
	img_1D=subAdTHimg.ravel()
	bincolorCount = Counter(img_1D)
	bincolorCountKEY=list(bincolorCount.keys())
	lowValue=255
	jj=0
	for i in range(1,len(bincolorCountKEY)):
		localV=bincolorCountKEY[i]
		sumColorV=sumColorV+bincolorCount[localV]*localV
		jj+=bincolorCount[localV]
		
	AvercolorV=int(sumColorV/jj*1.0)
	
	_,subAdTHimg = cv2.threshold(subAdTHimg, avershortcolor, 255, cv2.THRESH_BINARY_INV)
	pics.append(subAdTHimg)#20220307
	insideAreaH=int((inB*inB)*3.14*0.5)
	kernel33 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
	subAdTHimg=cv2.morphologyEx(subAdTHimg, cv2.MORPH_CLOSE, kernel33, iterations=1)

	contours,hierarchy = cv2.findContours(subAdTHimg, 1, 2)
	whRatioL=[]
	angleL=[]
	areaL=[]
	AA=[]
	sfV=[]
	j=0
	maybeMainscalesAngle={}
	tm=[]
	angleTolerance=12
	for i in range(len(contours)):
		area=cv2.contourArea(contours[i])
		
		angle, cntr, firstEigenvalue, secondEigenvalue=getOrientation(contours[i]) #firstEigenvalue, secondEigenvalue
		if (firstEigenvalue==0.0):
			continue
		scaleToCenterV=(center[0]-cntr[0], center[1]-cntr[1])
		angleStoC = degrees(atan2(scaleToCenterV[1], scaleToCenterV[0]))
		angleO=angle
		angleStoCO=round(angleStoC, 1)
		if(angle<0):
			angle+=180
		if(angleStoC<0):
			angleStoC+=180
		if(angleStoCO<0):
			angleStoCO=360+angleStoCO


		area1, whRatio1, compactness1, wlen1, hlen1, approx1=ccParameter(contours[i])
		

		if(-angleTolerance<(angle-angleStoC)<angleTolerance and shortScalesMean*1.3<area<insideAreaH and whRatio1<0.25): #shortScalesMedian*1.1
			sfvalue=round(firstEigenvalue/secondEigenvalue,3)
			angleStoCOoffset=round((angleStoCO+90)%360,1)
			maybeMainscalesAngle[angleStoCOoffset]=[cntr,angleStoCOoffset, contours[i]]
			sfV.append(sfvalue)
			AA.append(area)
			
			j+=1


	maybeMainscalesAngleLSort=sorted(list(maybeMainscalesAngle.keys()))
	maybeMainscalesAngleLSort.append(maybeMainscalesAngleLSort[0]+360)

	diffAngleL=[]
	for i in range(1,len(maybeMainscalesAngleLSort)):
		diffAngleL.append(maybeMainscalesAngleLSort[i]-maybeMainscalesAngleLSort[i-1])

	diffAngleM=[]
	for i in range(len(diffAngleL)):

		if (len(diffAngleM)==0):
			diffAngleM.append([diffAngleL[i]])
		else:
			booD=1
			for j in range(len(diffAngleM)):
				for k in range(-10, 11):
					if(diffAngleM[j][0]+k == diffAngleL[i]):
						diffAngleM[j].append(diffAngleL[i])
						booD=0
			if(booD==1):
				diffAngleM.append([diffAngleL[i]])

	t=len(diffAngleM[0])
	for i in range(len(diffAngleM)):
		if(len(diffAngleM[i])>t):
			t=len(diffAngleM[i])

	angleTH=0
	for i in range(len(diffAngleM)):
		if(len(diffAngleM[i]) == t):
			angleTH=diffAngleM[i][0]

	MainscalesAngleLSort=[]
	for i in range(len(maybeMainscalesAngleLSort)):
		if(maybeMainscalesAngleLSort[i] >= 360):
			break
		else:
			MainscalesAngleLSort.append(maybeMainscalesAngleLSort[i])
			cv2.drawContours(scalemarkClear,[maybeMainscalesAngle[maybeMainscalesAngleLSort[i]][-1]],0,(int(255)),-1) #maybeMainscalesAngle[angleStoCOoffset]
			cv2.circle(scalemarkClear, (int(maybeMainscalesAngle[maybeMainscalesAngleLSort[i]][0][0]), int(maybeMainscalesAngle[maybeMainscalesAngleLSort[i]][0][1])), 5, (128), -1)


	MainscalesAngleL=list(set(MainscalesAngleLSort))
	
	#=====2022 2 22 add
	MainscalesAngleLS=sorted(list(set(MainscalesAngleLSort)))
	diffAAL=[]
	for i in range(1, len(MainscalesAngleLS)):
		diffAAL.append(MainscalesAngleLS[i]-MainscalesAngleLS[i-1])
	diffAALmost=[]
	diffAALmostlen=1
	for i in range(len(diffAAL)):
		if(len(diffAALmost)==0):
			diffAALmost.append([diffAAL[i]])
		else:
			boo1=1
			for j in range(len(diffAALmost)):
				if(diffAALmost[j][0]-5<diffAAL[i]<diffAALmost[j][0]+5):
					diffAALmost[j].append(diffAAL[i])
					if(diffAALmostlen<len(diffAALmost[j])):
						diffAALmostlen=len(diffAALmost[j])
					boo1=0
			if(boo1==1):
				diffAALmost.append([diffAAL[i]])
	gapAnglemin=-1
	gapAngle=-1
	for i in range(len(diffAALmost)):
		if(len(diffAALmost[i])==diffAALmostlen):

			gapAngleM=min(diffAALmost[i])
			if(gapAnglemin==-1):
				gapAnglemin=gapAngleM
				gapAngle=diffAALmost[i][0]
			if(gapAngleM>gapAnglemin):
				gapAnglemin=gapAngleM
				gapAngle=diffAALmost[i][0]
			

	MainscalesAngleL_V3=[]
	for i in range(1, len(MainscalesAngleLS)):
		ss=MainscalesAngleLS[i]-MainscalesAngleLS[i-1]
		#print(ss, gapAngle)
		if(gapAngle-5<ss<gapAngle+5):
			MainscalesAngleL_V3.append(MainscalesAngleLS[i-1])
			MainscalesAngleL_V3.append(MainscalesAngleLS[i])
	MainscalesAngleL_V3=sorted(list(set(MainscalesAngleL_V3)))
	MainscalesAngleL_V4=MainscalesAngleL_V3.copy()
	for i in range(1, len(MainscalesAngleL_V3)):
		ss=MainscalesAngleL_V3[i]-MainscalesAngleL_V3[i-1]
		ss=ss/gapAnglemin
		if(ss>=2):
			for j in range(1,int(ss)):
				MainscalesAngleL_V4.append(MainscalesAngleL_V3[i-1]+gapAnglemin*j)
	MainscalesAngleL_V4=sorted(MainscalesAngleL_V4)

	MScalesL={}
	for i in range(len(MainscalesAngleLSort)):#MainscalesAngleL
		cv2.drawContours(scalemarks_t,[maybeMainscalesAngle[MainscalesAngleLSort[i]][2]],0,(int(255)),-1)
	for i in range(len(MainscalesAngleL_V3)):
		cv2.drawContours(scalemarks,[maybeMainscalesAngle[MainscalesAngleL_V3[i]][2]],0,(int(255)),-1)

	pics.append(scalemarks)#20220307
	scalesH=int((longAver+shortAver)/2)
	for i in range(len(MainscalesAngleL_V4)):
		if MainscalesAngleL_V4[i] in MainscalesAngleL_V3:
			maybeMainscalesAngle[MainscalesAngleL_V4[i]]
	for i in range(len(MainscalesAngleL_V4)):
		if MainscalesAngleL_V4[i] in MainscalesAngleL_V3:
			MScalesL[MainscalesAngleL_V4[i]]=maybeMainscalesAngle[MainscalesAngleL_V4[i]]
		else:
			MScalesL[MainscalesAngleL_V4[i]]=[None, None, None]
	pics.append(scalemarks_t)#20220307
	return scalemarkInside, pics, MScalesL, scalemarkClear, subAdTHimg
	
def main():
	pass


if __name__ == "__main__" :
	main()
