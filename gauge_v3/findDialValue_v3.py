import cv2
import numpy as np
import math
import keras_ocr
from matplotlib import pyplot as plt
from math import atan2, cos, sin, sqrt, pi, degrees

def textctrl(p4):
	tx=0
	ty=0
	for i in p4:
		tx+=i[0]
		ty+=i[1]
	return((int(tx/4), int(ty/4)))

def floatValueDetect(ocrWord, images):
	numV=int(ocrWord[0])
	img=images[int(ocrWord[1][0][1]):int(ocrWord[1][2][1]), int(ocrWord[1][0][0]):int(ocrWord[1][2][0])]
	allArea=img.shape[0]*img.shape[1]
	imgV=np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
	
	num_labels, labels,stats, centroids= cv2.connectedComponentsWithStats(img, connectivity = 8)
	output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
	c=0
	wordxL=[]
	for k in range(0, num_labels):
		if(stats[k][4]<allArea/2):
			cv2.rectangle(output, (stats[k][0], stats[k][1]), (stats[k][0]+stats[k][2], stats[k][1]+stats[k][3]), (0,0,255), 1, cv2.LINE_AA)
			if(stats[k][2]<stats[k][3]):
				c+=1
				wordxL.append(centroids[k][0])
	wordxL=sorted(wordxL)
	dotL=[]
	if(len(ocrWord[0])==c and c>1 ):
		for k in range(0, num_labels):
			if(wordxL[0]<centroids[k][0]<wordxL[-1] and stats[k][4]<allArea/2):
				dotL.append(k)
		dotID=-1
		dotArea=0
		for k in range(len(dotL)):
			if(dotID==-1):
				dotID=dotL[k]
				dotArea=stats[dotL[k]][4]
			if(stats[dotL[k]][4] > dotArea):
				dotID=dotL[k]
				dotArea=stats[dotL[k]][4]
		kk=0
		for k in range(1,len(wordxL)):
			if(wordxL[k-1]<centroids[dotID][0]<wordxL[k]):
				kk=k
				break
		numV=round(numV*(0.1**kk), kk)
	return numV

def testDialValue(pipeline, img):
	OCR_Region_Img=img.copy()
	color = (0,0,255)
	thickness = 2
	isClosed = True
	prediction_groups = pipeline.recognize([img])
	a=[]
	temp=[]
	sumV=0
	for i in range(len(prediction_groups[0])):
		if(prediction_groups[0][i][0].isdigit()):
			temp.append(len(str(prediction_groups[0][i][0]).replace('.','')))
	aver=int(sum(temp)/len(temp))

	for i in range(len(prediction_groups[0])):
		pts=np.around(prediction_groups[0][i][1])
		pts = pts.astype(int)
		pts = pts.reshape((-1, 1, 2))
		OCR_Region_Img = cv2.polylines(OCR_Region_Img, [pts], isClosed, color, thickness)
		if(prediction_groups[0][i][0]=='o'):
			Tctrl=textctrl(prediction_groups[0][i][1])
			a.append((0,Tctrl))
		if(prediction_groups[0][i][0].isdigit() and len(str(prediction_groups[0][i][0])) <=aver+1):
			Tctrl=textctrl(prediction_groups[0][i][1])
			fValue=floatValueDetect(prediction_groups[0][i],img)
			a.append((fValue,Tctrl))
	return a, OCR_Region_Img


def matchDailvalue(MScalesD, DVinfo, center):
	MSAngleL=sorted(list(MScalesD.keys()))
	MSAnglebool = np.ones((len(MSAngleL)), np.uint8)
	MSmatchD={}
	for i in range(len(MSAngleL)):
		MSmatchD[MSAngleL[i]]=None
	for i in range(len(DVinfo)):
		DVV=(center[0]-DVinfo[i][1][0], center[1]-DVinfo[i][1][1])
		DVAngle = round(degrees(atan2(DVV[1], DVV[0])),0)
		if(DVAngle<0):
			DVAngle=360+DVAngle
		DVAngle=round((DVAngle+90)%360,0)
		disDV=-1
		disDVid=-1
		for j in range(len(MSAngleL)):
			if(MScalesD[MSAngleL[j]][0]==None):
				continue
			if(-10<DVAngle-MSAngleL[j]<10 and MSAnglebool[j]==1):
				try:
					dis=((MScalesD[MSAngleL[j]][0][0]-DVinfo[i][1][0])*(MScalesD[MSAngleL[j]][0][0]-DVinfo[i][1][0]))+((MScalesD[MSAngleL[i]][0][1]-DVinfo[i][1][1])*(MScalesD[MSAngleL[i]][0][1]-DVinfo[i][1][1]))
					#print('dis', dis )
					if(disDV ==-1):
						disDV=dis
						disDVid=j
					if(dis<disDV):
						disDV=dis
						disDVid=j
				except:
					pass
		if(disDVid!=-1):
			MSmatchD[MSAngleL[disDVid]]=DVinfo[i][0]
			MSAnglebool[disDVid]=0

	#===================
	MSmatchValue=list(MSmatchD.values())
	#0以下加負號
	zeroID=-1
	for i in range(len(MSmatchValue)):
		if(MSmatchValue[i]==0):
			zeroID=i
			break
	if(zeroID>0):
		for i in range(zeroID):
			if(MSmatchValue[i]!=None):
				if(MSmatchValue[i]>0):
					MSmatchD[MSAngleL[i]]=-MSmatchValue[i]

	#未被判斷成浮點數的浮點數
	f=0
	fLsum=0
	for i in range(len(MSmatchValue)):
		if(isinstance(MSmatchValue[i], float) and int(MSmatchValue[i])!=MSmatchValue[i]):
			fLsum=fLsum+len(str(MSmatchValue[i]).split('.')[0])
			f+=1
	if(f>0 and f/len(MSmatchValue)>=0.5):
		fLaver=round((fLsum/f),0)
		for i in range(len(MSmatchValue)):
			if(MSmatchValue[i]!=None and MSmatchValue[i]>=10):
				MSmatchD[MSAngleL[i]]=MSmatchValue[i]*(0.1**fLaver)

	#異常數值
	j=1
	i=0
	gapValueL=[]
	gapValuelen=1
	while(i<len(MSmatchValue)-1):
		if(MSmatchValue[i]!=None):
			tt=1
			for j in range(i+1, len(MSmatchValue)):
				if(MSmatchValue[j]!=None):
					tN=(MSmatchValue[j]-MSmatchValue[i])/tt*1.0
					if(len(gapValueL)==0):
						gapValueL.append([tN])
					else:
						boo1=1
						for k in range(len(gapValueL)):
							if(gapValueL[k][0]==tN):
								gapValueL[k].append(tN)
								if(gapValuelen<len(gapValueL[k])):
									gapValuelen=len(gapValueL[k])
								boo1=0
						if(boo1==1):
							gapValueL.append([tN])
					break
				else:
					tt+=1
			i=j
		else:
			i+=1
	gapV=-1
	for i in range(len(gapValueL)):
		if(len(gapValueL[i])==gapValuelen):
			if(gapV==-1):
				gapV=gapValueL[i][0]
			if(gapV>gapValueL[i][0]):
				gapV=gapValueL[i][0]
	j=1
	i=0
	subCount=np.zeros(len(MSmatchValue))
	while(i<len(MSmatchValue)-1):
		if(MSmatchValue[i]!=None):
			tt=1
			for j in range(i+1, len(MSmatchValue)):
				if(MSmatchValue[j]!=None):
					tN=(MSmatchValue[j]-MSmatchValue[i])/tt*1.0
					if(tN==gapV):
						subCount[i]+=1
						subCount[j]+=1
					
					break
				else:
					tt+=1
			i=j
		else:
			i+=1
	sc=0
	FP=-1
	for i in range(len(subCount)):
		if(subCount[i]>0):
			sc+=1
	if(sc==2):
		for i in range(len(subCount)):
			if(subCount[i]>0):
				FP=i
				break
	if(sc>2):
		for i in range(len(subCount)):
			if(subCount[i]==2):
				FP=i
				break
	startNum=MSmatchD[MSAngleL[FP]]
	if(startNum!=None):
		for i in range(FP,len(MSAngleL)):
			MSmatchD[MSAngleL[i]]=startNum
			startNum+=gapV
		startNum=MSmatchD[MSAngleL[FP]]
		for i in range(FP, -1, -1):
			MSmatchD[MSAngleL[i]]=startNum
			startNum-=gapV
		MSmatchValue_v2=list(MSmatchD.values())
		Msbool=0
		for i in range(1,len(MSmatchValue_v2)):
			if(MSmatchValue_v2[i]<=MSmatchValue_v2[i-1]):
				Msbool=1
		if(Msbool==1):
			for i in range(len(MSAngleL)):
				MSmatchD[MSAngleL[i]]=None
	return MSmatchD

def gaugeR(mainScaleMarkD, offsetNiddleA, imgO, errorgapPNum):
	if(offsetNiddleA<0):
		offsetNiddleA=360+offsetNiddleA
	offsetAngle=round((offsetNiddleA+90)%360,0)
	Gaugevalue=None
	MSMkeys=list(mainScaleMarkD.keys())
	MSMgap=MSMkeys[1]-MSMkeys[0]
	times=1
	GVtype=-1
	errPth=0.06
	if(offsetAngle<MSMkeys[0] and errorgapPNum<errPth): #if(offsetAngle<MSMkeys[0]):
		disA=(offsetAngle-MSMkeys[0])/(MSMkeys[1]-MSMkeys[0])
		disV=mainScaleMarkD[MSMkeys[1]]-mainScaleMarkD[MSMkeys[0]]
		Gaugevalue=round(mainScaleMarkD[MSMkeys[0]]+disV*disA,2)
		GVtype=1
	elif(offsetAngle>=MSMkeys[-1] and errorgapPNum<errPth): #elif(offsetAngle>=MSMkeys[-1]):
		disA=(offsetAngle-MSMkeys[-1])/(MSMkeys[-1]-MSMkeys[-2])
		disV=mainScaleMarkD[MSMkeys[-2]]-mainScaleMarkD[MSMkeys[-1]]
		Gaugevalue=round(mainScaleMarkD[MSMkeys[-1]]+disV*disA,2)
		GVtype=2
	elif(MSMkeys[0]<offsetAngle<MSMkeys[-1]):#else:
		for i in range(len(MSMkeys)-1):
			if(MSMkeys[i]<=offsetAngle<MSMkeys[i+1]):
				disA=(offsetAngle-MSMkeys[i])/(MSMkeys[i+1]-MSMkeys[i])
				disV=mainScaleMarkD[MSMkeys[i+1]]-mainScaleMarkD[MSMkeys[i]]
				Gaugevalue=round(mainScaleMarkD[MSMkeys[i]]+disV*disA,2)
		GVtype=0
	return Gaugevalue, MSMkeys[0]-times*MSMgap, MSMkeys[-1]+times*MSMgap, GVtype
	
def main():
	pass

if __name__ == "__main__" :
	main()