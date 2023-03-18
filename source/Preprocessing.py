import os
import cv2
import imutils
import numpy as np
import math
import matplotlib.pyplot as plt
from segment import wordSegmentation


def alignImages(im1):

	MAX_FEATURES = 4000
	GOOD_MATCH_PERCENT = 0.5
	# Load image
	#im1 là img cần chỉnh sửa, im2 la anh mau
	# im1 = cv2.imread('doc/giaythi102.jpg', cv2.IMREAD_COLOR)
	im2 = cv2.imread('doc/giaythichuan.jpg', cv2.IMREAD_COLOR)

	# Convert images to grayscale
	im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
	im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

	# Detect ORB features and compute descriptors.
	orb = cv2.ORB_create(MAX_FEATURES)
	keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
	keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

	# Match features.
	matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
	matches = matcher.match(descriptors1, descriptors2, None)

	# Sort matches by score
	#matches.sort(key=lambda x: x.distance, reverse=False)
	matches = tuple(sorted(matches, key=lambda x: x.distance, reverse=False))
	# Remove not so good matches
	numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
	matches = matches[:numGoodMatches]

	# Draw top matches

	imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)

	# cv2.imwrite("doc/matches51.jpg", imMatches)

	# Extract location of good matches
	points1 = np.zeros((len(matches), 2), dtype=np.float32)
	points2 = np.zeros((len(matches), 2), dtype=np.float32)

	for i, match in enumerate(matches):
		points1[i, :] = keypoints1[match.queryIdx].pt
		points2[i, :] = keypoints2[match.trainIdx].pt

	# Find homography
	h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
	# Use homography
	height, width, channels = im2.shape
	im1Reg = cv2.warpPerspective(im1Gray, h, (width, height))
	im1Reg = maximizeContrast(im1Reg)
	# cv2.imshow('im1Reg',im1Reg)
	return im1Reg, height, width


### For image #####
def imformation_crop(im1):  
	(im1Reg, height, width) = alignImages(im1)

# 		########
	MSSV_crop = im1Reg[315 : 70 + 300, 260 : 300 + 400]
	name_crop = im1Reg[230 : 80 + 250, 305 : 350 + 580]
	diem_crop = im1Reg[568 : 380 + 430, 95 : 200 + 210]
 
 #name_crop = im1Reg[255 : 80 + 250, 305 : 300 + 580]
 #diem_crop = im1Reg[570 : 380 + 430, 100 : 200 + 210]
# 	# diem_crop = im1Reg[590 : 380 + 400, 140 : 200 + 190]
	# cv2.imshow('mssv',MSSV_crop)
	# cv2.waitKey(0)
	# cv2.imwrite("doc/mssv_crop3.jpg", MSSV_crop)
	# cv2.imwrite("doc/ten_crop1.jpg", name_crop)
	# cv2.imwrite("doc/diem_crop1.jpg", diem_crop)
	# cv2.imshow('name',name_crop)
	# cv2.waitKey(0)
	# cv2.imshow('score',diem_crop)
	# cv2.waitKey(0)
	return MSSV_crop, name_crop, diem_crop

def removeline(image):
	
	thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
	# cv2.imshow('thresh', thresh)

	# Remove horizontal
	horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,1))
	detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
	# cv2.imshow('detected_lines', detected_lines)
	cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	for c in cnts:
		cv2.drawContours(image, [c], -1, (255,255,255), 2)
	# cv2.imshow('image213', image)
	# cv2.waitKey(0)
	repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,3))
	result = 255 - cv2.morphologyEx(thresh - detected_lines, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
	
	return result
	




def removecircle(image):
	thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
	result = 255 - thresh
	
	# # Remove horizontal
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
	detected_lines = cv2.dilate(thresh,kernel,iterations=1)
	# cv2.imshow('detected_lines', detected_lines)
	cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
	for c in cnts:
		cv2.drawContours(result, [c], -1, (255,255,255), 10)

	# #Extract real score
	img = 255 - result
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
	imgThres = cv2.dilate(img,kernel,iterations=2)
	# cv2.imshow('imgThresgeg',imgThres)
	score = result
	# cv2.imshow('score11',score)
 
	# find connected components
	cnts = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
	for c in cnts:
		currBox = cv2.boundingRect(c)
		(x, y, w, h) = currBox
		score = result[y:y+h, x:x+w]
		# cv2.imshow('score',score)
	return score


def maximizeContrast(imgGrayscale):
	#Làm cho độ tương phản lớn nhất 
	height, width = imgGrayscale.shape
	
	imgTopHat = np.zeros((height, width, 1), np.uint8)
	imgBlackHat = np.zeros((height, width, 1), np.uint8)
	structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) #tạo bộ lọc kernel
	
	imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement, iterations = 6) #nổi bật chi tiết sáng trong nền tối
	# cv2.imwrite("tophat.jpg",imgTopHat)
	imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement, iterations = 6) #Nổi bật chi tiết tối trong nền sáng
	# cv2.imwrite("blackhat.jpg",imgBlackHat)
	imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat) 
	imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

	# cv2.imshow("imgGrayscalePlusTopHatMinusBlackHat",imgGrayscalePlusTopHatMinusBlackHat)
	#Kết quả cuối là ảnh đã tăng độ tương phản 

	return imgGrayscalePlusTopHatMinusBlackHat    



def main():
	# img  = cv2.imread('data\Class_list_constrained\giaythi55.jpg', cv2.IMREAD_COLOR)
	# MSSV_crop, name_crop, diem_crop = imformation_crop(img)
	# diem_crop_copy = diem_crop.copy()
	# diem_crop_copy = removecircle(diem_crop_copy)
	# cv2.imshow('after line', diem_crop)
	# cv2.imshow('after line', name_crop)
	 
	if not os.path.exists('doc/removeline_word'):
		os.mkdir('doc/removeline_word')
	for filename in os.listdir("data\Class_list_constrained"):
		print ('filename',filename)
		filepath  = os.path.join("data\Class_list_constrained"+ '/', filename)
		img  = cv2.imread(filepath, cv2.IMREAD_COLOR)
		MSSV_crop, name_crop, diem_crop = imformation_crop(img)

		name_crop = removeline(name_crop)
		MSSV_crop = removeline(MSSV_crop)
		# diem_crop = removecircle(diem_crop)
  
		result = wordSegmentation(name_crop, kernelSize=21, sigma=11, theta=4, minArea=500)
		name_recognized = str()
		draw = []
		i = 0
		for line in result:
			if len(line):
				for (_, w) in enumerate(line):
					(wordBox, wordImg) = w


					i = i+1
					imwritepath = os.path.join('doc/removeline_word/' + filename[:-4]+str(i)+'.png')
					cv2.imwrite(imwritepath, wordImg)
		# cv2.imshow('name_crop', name_crop)
		# cv2.imwrite("doc/removeline_word/namecrop_giaythi1.jpg",name_crop)
		
		# cv2.imshow('mssv', MSSV_crop)
		# cv2.imwrite("doc/mssv_crop3_2.jpg",MSSV_crop)
		# cv2.imshow('diem', diem_crop)
		# cv2.imwrite("doc/removecircle/diemcrop_giaythi1.jpg",diem_crop)
		# cv2.waitKey(0)
	cv2.waitKey(0)

if __name__ =='__main__':
	main()
