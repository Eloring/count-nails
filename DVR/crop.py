import cv2
import os
import numpy as np
import collections
import projection
from glob import glob
from PIL import Image
from pytesseract import image_to_string
import time
from pyExcelerator import *
import json 
import sys

def hsvDict():
	dict = collections.defaultdict(list)

	# red
	lower_red = np.array([156, 43, 46])
	upper_red = np.array([180, 255, 255])
	color_list = []
	color_list.append(lower_red)
	color_list.append(upper_red)
	dict['red']=color_list
 
	# red
	lower_red = np.array([0, 100, 100])
	upper_red = np.array([10, 255, 255])
	color_list = []
	color_list.append(lower_red)
	color_list.append(upper_red)
	dict['red2'] = color_list

	# blue
	lower_blue = np.array([78, 43, 46])
	upper_blue = np.array([124, 255, 255])
	color_list = []
	color_list.append(lower_blue)
	color_list.append(upper_blue)
	dict['blue'] = color_list

    	return dict

def cropBlock(image):
	hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
	grayimg = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	color_dict = hsvDict()
	mask = cv2.inRange(hsv,color_dict['red2'][0],color_dict['red2'][1])
	bimg = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
	kernel = np.ones((3,3), np.uint8)
	opening = cv2.morphologyEx(bimg, cv2.MORPH_OPEN, kernel, iterations = 3)
	erosion = cv2.erode(opening, kernel, iterations = 2)
	# cv2.imwrite('redpoint.jpg', erosion)
	cnts = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	points = []
	for i in range(len(cnts[1])):
           
		points.append(cnts[1][i][-3][0])
	points = np.array(points)
	points = points[np.lexsort(points[:,::-1].T)]
	image_list = collections.defaultdict(list)
	for j in range(len(points)-1):
		img_list=[]
		x1,y1=points[j]
		x2,y2=points[j+1]
		if y1<y2:
			img = grayimg[y1:y2,x1:x2]
		else:
			img = grayimg[y2:y1,x1:x2]
		# img = cv2.equalizeHist(img)
		img = np.uint8(np.clip((0.9*img+1),0,255))	
		img_list.append(img)
        	image_list[COLORLIST[j]] = img_list
		# cv2.imwrite('result/'+str(j)+'.jpg',img)
	return image_list

def cropBlock2(image):
	hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
	grayimg = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	color_dict = hsvDict()
	mask = cv2.inRange(hsv,color_dict['red2'][0],color_dict['red2'][1])
	bimg = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
	kernel = np.ones((3,3), np.uint8)
	opening = cv2.morphologyEx(bimg, cv2.MORPH_OPEN, kernel, iterations = 3)
	erosion = cv2.erode(opening, kernel, iterations = 2)
	# cv2.imwrite('redpoint.jpg', erosion)
	cnts = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	points = []
	for i in range(len(cnts[1])):
		points.append(cnts[1][i][-3][0])
	points = np.array(points)
	points = points[np.lexsort(points[:,::-1].T)]
	image_list = collections.defaultdict(list)
	for j in range(len(points)-1):
		img_list=[]
		x1,y1=points[j]
		x2,y2=points[j+1]
		if y1<y2:
			img = grayimg[y1:y2,x1:x2]
		else:
			img = grayimg[y2:y1,x1:x2]
		# img = cv2.equalizeHist(img)
		img = np.uint8(np.clip((0.9*img+1),0,255))	
		img_list.append(img)
        	image_list[COLORLIST[j]] = img_list
		# cv2.imwrite('result/'+str(j)+'.jpg',img)
	return image_list

def cropLineByPro(color, img,template):
	a = np.array(projection.hor_projection(img), dtype = np.int64)
	lineList = []
	print "=====",color,len(a), a
	for i in range (len(a)-1):
		start = a[i]
		end = a[i+1]
		while (end-start)<template.shape[0] :	
			if start >0:
				start = start-5
			if end < (img.shape[0]-5):
				end = end+5
		lineList.append(img[start:end,:])
		# cv2.imwrite(str(i)+'_line.jpg', lineList[i])
	return lineList
def findBox(boxName):
	f = open(FILE_HOME+'/src/DVR/boxTemp.json')
	box = json.load(f)
	global boxDict,COLORLIST,LINELIST,HOLELIST
	boxDict = box[boxName]
	COLORLIST = []
	LINELIST = []
	HOLELIST = []
	for i in boxDict:
                print i
		key = str(i.keys()[0])
		COLORLIST.append(key)
                print i[key][1] 
		HOLELIST.append(int(i[key][1]))
                print i[key][0]
		LINELIST.append(int(i[key][0]))
	
def cropLineByPer(lineNum, img, template):
	lineHeight = int(img.shape[0]/lineNum)
	lineList = []
	start = 0
	for i in range(0, lineNum):
		lineList.append(img[start:(start+lineHeight),])
		start += lineHeight
	return 	lineList
		
	
	
def temMatch(line, template):
	w,h = template.shape
	res = cv2.matchTemplate(line,template,5)
        print res
	threshold = 0.7
	loc = np.where(res >= threshold)
	holesNum = 0
	last_x = 0
	x_loc = loc[1]
	x_loc = sorted(x_loc)
	for x in x_loc :
		if x - last_x  > h:
			holesNum +=1
			last_x = x
	return holesNum

def temMatchForUCS(line, template):
	w,h = template.shape
	res = cv2.matchTemplate(line,template,5)
	
	threshold = 0.6
	loc = np.where(res >= threshold)
	holesNum = 0
	last_y = 0
	y_loc = loc[0]
	y_loc = sorted(y_loc)
	for y in y_loc :
		if y - last_y  > w/2:
			holesNum +=1
			last_y = y
	return holesNum

def numRec(line):
	num_files = glob('./num_temp/*.jpg')
	numName = './num_temp/Unknown.jpg'
	for numTemName in num_files:
		numTem = cv2.imread(numTemName,0)
		w,h = numTem.shape
		if (line.shape[0]>w and line.shape[1]>h):
			res = cv2.matchTemplate(line,numTem,5)
			threshold = 0.9
			loc = np.where(res >= threshold)
			if (len(loc[1])):
				numName = numTemName
				# for pt in zip(*loc[::-1]):
	    				# cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (7,249,151), 2)
				# cv2.imwrite(numTemName+'_result.jpg',img)
	fn = os.path.basename(numName)
	return fn[:-4]

def numRec2(line):
	image = Image.fromarray(line)
	num = 'unknown'
	num = image_to_string(image, lang='eng', boxes=False, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
	return num

def create_fold(path):
	folder = os.path.exists(path)
	if not folder:
		os.makedirs(path)  
def txt2excel(txtName):
	count = []
	with open(txtName, 'r') as f:
		for line in f:
			if line.strip()=='':
				continue
			count.append(str(line.strip('\n')))
		f.close()
	TypeName = ['FP10', 'FP12', 'FP12', 'FP14', 'FP14', 'FP16', 'FP16', 'FP18', 'FP18', 
				'FP20', 'FP22', 'FP24', 'FP26', 'FP28', 'FP30', 'SP10', 'SP12', 'SP12', 
				'SP14', 'SP14', 'SP16', 'SP16', 'SP18', 'SP18', 'SP20', 'SP22', 'SP24', 
				'SP26', 'SP28', 'SP30', 'CSL10', 'CSL11', 'CSL12', 'CSL13', 'CSL14', 
				'CSL15', 'CSL16', 'CSL18', 'CSL20', 'CS10', 'CS11', 'CS12', 'CS13', 
				'CS14', 'CS15', 'CS16', 'CS18', 'CS20', 'USC', 'MDTP14', 'MDTP16', 
				'MDTP18', 'MDTP20', 'MDTP20', 'MDTP22', 'MDTP22', 'MDTP24', 'MDTP24', 
				'MDTP26', 'MDTP26', 'MDTP28', 'MDTP30', 'TP10', 'TP12', 'TP14', 'TP16', 
				'TP16', 'TP18', 'TP18', 'TP20', 'TP20', 'TP22', 'TP22', 'TP24', 'TP24', 
				'TP26', 'TP28', 'TP30', 'P10', 'P12', 'P14', 'P16', 'P16', 'P18', 'P18', 
				'P20', 'P20', 'P22', 'P22', 'P24', 'P24', 'P26', 'P28', 'P30',]

	for i in count:
		if len(i) > 2:
			count.remove(i)
	workbook = Workbook()
	worksheet = workbook.add_sheet('sheet1')
	worksheet.write(0,0,'Type')
	worksheet.write(0,1,'Number')

	for j in range(0,len(count)):
		worksheet.write(j+1,0,TypeName[j])
		worksheet.write(j+1,1,count[j])
	workbook.save(txtName[:-4]+'.xls') 

def box_DVR(image,boxName):
	global FILE_HOME
	FILE_HOME = sys.path[0]
	TOTAL = 0
	findBox(boxName)
	num_list=[]
	# image = cv2.imread('DVR.jpg', 1)
	block_list = cropBlock(image)
	for j in range(len(COLORLIST)) :
		color = COLORLIST[j]
		template = cv2.imread(FILE_HOME+'/src/DVR/template/'+boxName+'/'+color+'_tem.jpg',0)
		block = block_list[color][0]
		
		if (color == 'UCS'):
			hole_num = temMatchForUCS(block, template)
			nail_num = int(LINELIST[j])-hole_num
			TOTAL += nail_num
			num_list.append(nail_num)
		else:
			lineNum = LINELIST[j]
			lineList = cropLineByPer(lineNum, block,template)
			num_block =0	
			for i in range(len(lineList)) :
				line = lineList[i]
				hole_num = temMatch(line, template)
				nail_num = int(HOLELIST[j])-hole_num
				TOTAL += nail_num
				num_block += nail_num
			num_list.append(num_block)
	
	return str(num_list) +'='+ str(TOTAL)
if __name__ == '__main__':
	

	boxName = 'DVR'
	findBox(boxName)
	image = cv2.imread('DVR.jpg', 1)

	'''
	block_list = cropBlock(image)
	now = time.strftime('%Y%m%d%H%M',time.localtime(time.time()))
	file_name = str(boxName)+'_'+str(now)
	f = open("result/"+file_name+".txt",'a')
	f.write(str(boxName))
	f.write("\n")
	# nail_ddta=DataFrame(index=COLORLIST,columns=NUMTEM)
	for j in range(len(COLORLIST)) :
		f.write("\n")
		color = COLORLIST[j]
		template = cv2.imread('template/'+boxName+'/'+color+'_tem.jpg',0)
		block = block_list[color][0]
		fold_path = 'result/'+color
		create_fold(fold_path)
		if (color == 'UCS'):
			print color
			hole_num = temMatchForUCS(block, template)
			f.write(str(hole_num))
			f.write("\n")
			cv2.imwrite(fold_path+'/'+str(hole_num)+'.jpg',block)
		else:
			lineNum = LINELIST[j]
			lineList = cropLineByPer(lineNum, block,template)	
			for i in range(len(lineList)) :
				line = lineList[i]
				hole_num = temMatch(line, template)
				f.write(str(hole_num))
				f.write("\n")
				# nail_len = numRec(line)
				cv2.imwrite(fold_path+'/'+str(i)+'_'+str(hole_num)+'.jpg',line)
	f.close()	
	txt2excel('result/'+file_name+'.txt')
	'''


