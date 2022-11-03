import sys
import cv2
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import pandas as pd
import csv
sys.path.insert(0, './gauge_v3')
from guagePreprocessing_v3 import preprocessing_passRW
from scaleMarkContour_v3 import findScaleMarksContour, subRegion, countScalsColorValue
from gaugeCenterCoordinate_v3 import gaugeCenterPoint, GC_CC
from scaleToCenter_v3 import scaleToCenter, scaleMarks
from gaugeNiddle_v3 import findGaugeNiddle
from findDialValue_v3 import testDialValue, matchDailvalue, gaugeR
from record_v3 import *
from makedata_v3 import *

def GRApi(guageImg, name,pipeline ,gaugetype):
	Gaugevalue=None
	errorgap=None
	avershortcolor=None
	offsetNiddleA=None
	errorgapPNum=-1
	newMSMD={}
	srange=lrange=-1
	GVtype=-1
	N=3

	guageImgcopy=guageImg.copy()
	scales=guageImg.copy()

	#前處理
	grayGuageImg = cv2.cvtColor(guageImg, cv2.COLOR_BGR2GRAY)
	binGI=preprocessing_passRW(grayGuageImg)
	
	#找短刻度線
	onlyScalemarkImg, CL, shortScaleAreaL= findScaleMarksContour(binGI)
	onlyScalemarkImgcopy=onlyScalemarkImg.copy()

	#儀表板中心
	clearImg2, topNMeanCenter=gaugeCenterPoint(onlyScalemarkImg,CL[0], guageImg, N)
	try:
		#get d1 and d2
		averStoC_D, shortAver, longAver, c5=scaleToCenter(CL[0], topNMeanCenter, onlyScalemarkImg)
		
		#get scale marks aver color
		avershortcolor=countScalsColorValue(grayGuageImg, onlyScalemarkImg,topNMeanCenter,  longAver, shortAver)

		#obtain main scale marks
		scales,scalemarks, MScalesL, scalemarkClear, subAdTHimg=subRegion(grayGuageImg, shortAver, longAver, topNMeanCenter, shortScaleAreaL, avershortcolor, onlyScalemarkImg,guageImgcopy) #binGIColor)
		
		#obtain dial values
		#DVinfo, OCR_Region_Img=testDialValue(pipeline, scales)
		
		#obtain main scale marks  and dial values binding 
		#MSmatchD=matchDailvalue(MScalesL, DVinfo, topNMeanCenter)

		#obtain niddle
		offsetNiddleAngle, binImgROI=findGaugeNiddle(binGI, CL[1], topNMeanCenter, averStoC_D, grayGuageImg, guageImgcopy, shortAver, longAver)

		#main scale marks  and dial values binding record
		# 0 = proposed method, 1 = min&max value
		# 0=man1, 1=man2, 2=man3, 3=acr2 , 4=類似man2
		MSmatchD=MSmatchDRec(0,gaugetype)
		MSmatchKeysL=list(MSmatchD.keys())
		MScalesLL=list(MScalesL.keys())

		halfgap=computegap(MSmatchKeysL)
		##guageImg=drawMSM(guageImg,topNMeanCenter, MSmatchKeysL, (0,0,255))
		##guageImg=drawMSM(guageImg,topNMeanCenter, MScalesLL, (255,0,0))
		newMSMD, newMSMgap, newMSMDKeysdiff=newMSM(MSmatchD, MScalesLL)
		errorgap, errorgapL, errorgapP, DVLL=errorGap(MSmatchKeysL, MScalesLL, halfgap)
		errorgapPNum=np.sum(errorgapP)/len(errorgapP)

		#gauge reading
		if(offsetNiddleAngle!=None and len(newMSMD)>1):
			#offsetNiddleA=(offsetNiddleAngle+90)%360
			Gaugevalue, srange, lrange, GVtype=gaugeR(newMSMD, offsetNiddleAngle, guageImg, errorgapPNum)
		else:
			Gaugevalue=None
		print("Gaugevalue", Gaugevalue)
		if(Gaugevalue!=None):
			if(isinstance(Gaugevalue, float)):
				fstr=str(Gaugevalue).split('.')
				GVstr=fstr[0]+'_'+fstr[1]
			else:
				GVstr=str(Gaugevalue)
		else:
			GVstr='None'
	except:
		pass

	return Gaugevalue

def main():
	pass
	

if __name__ == "__main__" :
	main()

