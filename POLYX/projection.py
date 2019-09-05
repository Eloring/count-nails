import sys
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

import cv2
import numpy as np
from scipy.signal import argrelextrema
from matplotlib import pyplot
from scipy.ndimage import measurements

def smooth(y, box_pts):
	# b = a
	# for i in range(3, len(a)-3):
	# 	b[i] = (a[i-2]+a[i-1]+a[i]+a[i+1]+a[i+2])
	# return b
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def findValley(a):
	a_smooth = np.asarray(smooth(a, 20))
	pyplot.plot(a_smooth)
	pyplot.show()
	valleys_index = []
	valleys_index.append(0)
	for i in range(1, len(a_smooth)-1):
		if (a_smooth[i-1]>=a_smooth[i]<a_smooth[i+1]):
			valleys_index.append(i)
	print valleys_index
	return valleys_index

def ver_projection(img):
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret,thresh1=cv2.threshold(img,130,255,cv2.THRESH_BINARY)
	(h,w)=img.shape
	# emptyImage = np.zeros((h,w), np.uint8)
	a = [0 for z in range(0, w)] 
	 
	for j in range(0,w):
	    for i in range(0,h): 
	        if  img[i,j]==255:
	            a[j]+=1 

	x_valleys = findValley(a)

	# for j in range(0,w):  
	#     for i in range((h-a[j]),h): 
	#         emptyImage[i,j]=255

	# cv2.imwrite("result/ver_projection.jpg", emptyImage)
	return x_valleys

def hor_projection(img):

	ret,img=cv2.threshold(img,130,255,cv2.THRESH_BINARY)
	cv2.imwrite("bimg.jpg", img)
	(h,w)=img.shape
	# emptyImage = np.zeros((h,w), np.uint8)
	a = [0 for z in range(0, h)] 
	 
	for i in range(0,h):
	    for j in range(0,w): 
	        if  img[i,j]==0:
	            a[i]+=1  	 


	y_valleys = findValley(a)
	# for i  in range(0,h):  
	#     for j in range(0,a[i]): 
	#         emptyImage[i,j]=255

	# cv2.imwrite("result/hor_projection.jpg", emptyImage)
	return y_valleys

def seg_img(img):
	x_valleys = np.array(ver_projection(img), dtype = np.int64)
	y_valleys = np.array(hor_projection(img), dtype = np.int64)
	for x in x_valleys:
		img[:,x:x+3]=0
	for y in y_valleys:
		img[y:y+3,:]=0
	# count numbers
	labels, nbr_objects = measurements.label(img)
	cv2.imwrite("result/seg_img.jpg", img)

if __name__ == '__main__':
	img = cv2.imread('input/y2.jpg',0)
	a = np.array(hor_projection(img), dtype = np.int64)
	# print a[len(a)-1]
	# print img.shape
	line = []
	for i in range (len(a)-1):
		line.append(img[a[i]:a[i+1],:])
		cv2.imwrite(str(i)+'_line.jpg', line[i])