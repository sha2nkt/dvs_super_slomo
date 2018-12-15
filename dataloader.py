import torch
import torch.utils.data as data
import numpy as np
import cv2
from PIL import Image
import glob
import os, pdb
import random


def populateTrainList(folderPath):
	folderList_pre = [x[0] for x in os.walk(folderPath)]
	folderList = []
	trainList = []

	for folder in folderList_pre:
		if folder[-3:] == '240':
			folderList.append(folder + "/" + folder.split("/")[-2])


	for folder in folderList:
		imageList = sorted(glob.glob(folder + '/' + '*.jpg'))
		for i in range(0, len(imageList), 12):
			tmp = imageList[i:i+12]
			if len(tmp) == 12:
			    trainList.append(imageList[i:i+12])

	
	return trainList


def populateTrainList2(rgbPath, dvsPath):
	folderListRgb = [x[0] for x in os.walk(rgbPath)]
	folderListDvs = [x[0] for x in os.walk(dvsPath)]
	trainList_rgb = []
	trainList_dvs = []

	for im_folder, dvs_folder in zip(folderListRgb, folderListDvs):
		imageList = sorted(glob.glob(im_folder + '/' + '*.jpg'))
		dvsList = sorted(glob.glob(dvs_folder + '/' + '*.png'))

		minLen = min(len(imageList), len(dvsList))
		imageList = imageList[:minLen]
		dvsList = dvsList[:minLen]

		for i in range(0, len(imageList), 12):
			tmp_rgb = imageList[i:i+12]
			tmp_dvs = dvsList[i:i+12]
			if len(tmp_rgb) == 12:
			    trainList_rgb.append(imageList[i:i+12])
			    trainList_dvs.append(dvsList[i:i+12])

	return trainList_rgb, trainList_dvs

def populateTrainList2Rgb(folderPath):
	folderList = [x[0] for x in os.walk(folderPath)]
	trainList = []

	for folder in folderList:
		imageList = sorted(glob.glob(folder + '/' + '*.jpg'))

		for i in range(0, len(imageList), 12):
			tmp = imageList[i:i+12]
			if len(tmp) == 12:
			    trainList.append(imageList[i:i+12])
	return trainList

def populateValList2Rgb(folderPath):
	folderList = [x[0] for x in os.walk(folderPath)]
	trainList = []

	for folder in folderList:
		imageList = sorted(glob.glob(folder + '/' + '*.jpg'))

		for i in range(0, len(imageList), 9):
			tmp = imageList[i:i+9]
			if len(tmp) == 9:
			    trainList.append(imageList[i:i+9])
	return trainList

# def populateDvsList2(folderPath):
# 	folderList = [x[0] for x in os.walk(folderPath)]
# 	trainList = []

# 	for folder in folderList:
# 		imageList = sorted(glob.glob(folder + '/' + '*.jpg'))
# 		for i in range(0, len(imageList), 12):
# 			tmp = imageList[i:i+12]
# 			if len(tmp) == 12:
# 			    trainList.append(imageList[i:i+12])
# 	return trainList

def populateTestList2(folderPath):
	folderList = [x[0] for x in os.walk(folderPath)]
	trainList = []

	for folder in folderList:
		imageList = sorted(glob.glob(folder + '/' + '*.jpg'))
		for i in range(0, len(imageList), 2):
			tmp = imageList[i:i+2]
			if len(tmp) == 2:
			    trainList.append(imageList[i:i+2])
	return trainList

def fixedCropOnList(image_list, output_size):
	cropped_img_list = []

	h,w = output_size
	height, width, _ = image_list[0].shape

	i = 100
	j = 0

	st_y = 0
	ed_y = w
	st_x = 0
	ed_x = h

	or_st_y = i 
	or_ed_y = i + w
	or_st_x = j
	or_ed_x = j + h    

	#print(st_x, ed_x, st_y, ed_y)
	#print(or_st_x, or_ed_x, or_st_y, or_ed_y)


	for img in image_list:
		new_img = np.empty((h,w,3), dtype=np.float32)
		new_img.fill(128)
		new_img[st_y: ed_y, st_x: ed_x, :] = img[or_st_y: or_ed_y, or_st_x: or_ed_x, :].copy()
		cropped_img_list.append(np.ascontiguousarray(new_img))


	return cropped_img_list


def randomCropOnList(image_list, output_size):
	
	cropped_img_list = []

	h,w = output_size
	height, width, _ = image_list[0].shape

	#print(h,w,height,width)

	i = random.randint(0, height - h)
	j = random.randint(0, width - w)

	st_y = 0
	ed_y = w
	st_x = 0
	ed_x = h

	or_st_y = i 
	or_ed_y = i + w
	or_st_x = j
	or_ed_x = j + h    

	#print(st_x, ed_x, st_y, ed_y)
	#print(or_st_x, or_ed_x, or_st_y, or_ed_y)


	for img in image_list:
		new_img = np.empty((h,w,3), dtype=np.float32)
		new_img.fill(128)
		new_img[st_y: ed_y, st_x: ed_x, :] = img[or_st_y: or_ed_y, or_st_x: or_ed_x, :].copy()
		cropped_img_list.append(np.ascontiguousarray(new_img))


	return cropped_img_list

def randomCropOnListDvs(image_list, dvs_list, output_size):
	
	cropped_img_list = []
	cropped_dvs_list = []

	h,w = output_size
	height, width, _ = image_list[0].shape
	dvs_h, dvs_w = dvs_list[0].shape

	#print(h,w,height,width)

	i = random.randint(0, height - h)
	j = random.randint(0, width - w)

	st_y = 0
	ed_y = w
	st_x = 0
	ed_x = h

	or_st_y = i 
	or_ed_y = i + w
	or_st_x = j
	or_ed_x = j + h    

	#print(st_x, ed_x, st_y, ed_y)
	#print(or_st_x, or_ed_x, or_st_y, or_ed_y)


	for img,dvs in zip(image_list, dvs_list):
		new_img = np.empty((h,w,3), dtype=np.float32)
		new_dvs = np.empty((h,w), dtype=np.float32)
		new_img.fill(128)
		new_img[st_y: ed_y, st_x: ed_x, :] = img[or_st_y: or_ed_y, or_st_x: or_ed_x, :].copy()
		new_dvs[st_y: ed_y, st_x: ed_x] = dvs[or_st_y: or_ed_y, or_st_x: or_ed_x].copy()
		cropped_img_list.append(np.ascontiguousarray(new_img))
		cropped_dvs_list.append(np.ascontiguousarray(new_dvs))

	return cropped_img_list, cropped_dvs_list


#print(len(populateTrainList('/home/user/data/nfs/')))

class expansionLoader(data.Dataset):

	def __init__(self, folderPath, mode='train'):

		self.trainList = populateTrainList2Rgb(folderPath)
		self.mode = mode
		print("# of training samples:", len(self.trainList))


	def __getitem__(self, index):

		img_path_list = self.trainList[index]
		start = random.randint(0,3)
		h,w,c = cv2.imread(img_path_list[0]).shape

		image = cv2.imread(img_path_list[0])
		
		#print(h,w,c)

		if h > w:
			scaleX = int(360*(h/w))
			scaleY = 360
		elif h <= w:
			scaleX = 360
			scaleY = int(360*(w/h))



		img_list = []

		flip = random.randint(0,1)
		if flip:
			for img_path in img_path_list[start:start+9]:
				tmp = cv2.resize(cv2.imread(img_path), (scaleX,scaleY))[:,:,(2,1,0)]
				img_list.append(np.array(cv2.flip(tmp,1), dtype=np.float32))
		else:
			for img_path in img_path_list[start:start+9]:
				tmp = cv2.resize(cv2.imread(img_path), (scaleX, scaleY))[:,:,(2,1,0)]
				img_list.append(np.array(tmp,dtype=np.float32))
		#cv2.imshow("j",tmp)
		#cv2.waitKey(0) & 0xff
		#brak
		for i in range(len(img_list)):
			#print(img_list[i].shape)
			#brak
			img_list[i] /= 255
			img_list[i][:,:,0] -= 0.485#(img_list[i]/127.5) - 1
			img_list[i][:,:,1] -= 0.456
			img_list[i][:,:,2] -= 0.406

			img_list[i][:,:,0] /= 0.229
			img_list[i][:,:,1] /= 0.224
			img_list[i][:,:,2] /= 0.225


		cropped_img_list = randomCropOnList(img_list,(352,352))

		for i in range(len(cropped_img_list)):
			cropped_img_list[i] = torch.from_numpy(cropped_img_list[i].transpose((2, 0, 1)))

		
		return cropped_img_list  

	def __len__(self):
		return len(self.trainList)


class testLoader(data.Dataset):

	def __init__(self, folderPath, mode='test'):

		self.trainList = populateTestList2(folderPath)
		self.mode = mode
		print("# of training samples:", len(self.trainList))


	def __getitem__(self, index):

		img_path_list = self.trainList[index]
		start = 0
		h,w,c = cv2.imread(img_path_list[0]).shape

		image = cv2.imread(img_path_list[0])
		
		#print(h,w,c)

		if h > w:
			scaleX = int(360*(h/w))
			scaleY = 360
		elif h <= w:
			scaleX = 360
			scaleY = int(360*(w/h))



		img_list = []

		for img_path in img_path_list[start:start+2]:
			tmp = cv2.resize(cv2.imread(img_path), (scaleX, scaleY))[:,:,(2,1,0)]
			img_list.append(np.array(tmp,dtype=np.float32))
		#cv2.imshow("j",tmp)
		#cv2.waitKey(0) & 0xff
		#brak
		for i in range(len(img_list)):
			#print(img_list[i].shape)
			#brak
			img_list[i] /= 255
			img_list[i][:,:,0] -= 0.485#(img_list[i]/127.5) - 1
			img_list[i][:,:,1] -= 0.456
			img_list[i][:,:,2] -= 0.406

			img_list[i][:,:,0] /= 0.229
			img_list[i][:,:,1] /= 0.224
			img_list[i][:,:,2] /= 0.225


		cropped_img_list = fixedCropOnList(img_list, (352, 352))		
		for i in range(len(cropped_img_list)):
			cropped_img_list[i] = torch.from_numpy(cropped_img_list[i].transpose((2, 0, 1)))

		
		return cropped_img_list 

	def __len__(self):
		return len(self.trainList)

class valLoader(data.Dataset):

	def __init__(self, folderPath, mode='test'):

		self.trainList = populateValList2Rgb(folderPath)
		self.mode = mode
		print("# of training samples:", len(self.trainList))


	def __getitem__(self, index):

		img_path_list = self.trainList[index]
		start = 0
		h,w,c = cv2.imread(img_path_list[0]).shape

		image = cv2.imread(img_path_list[0])
		
		#print(h,w,c)

		if h > w:
			scaleX = int(360*(h/w))
			scaleY = 360
		elif h <= w:
			scaleX = 360
			scaleY = int(360*(w/h))



		img_list = []

		for img_path in img_path_list[start:start+9]:
			tmp = cv2.resize(cv2.imread(img_path), (scaleX, scaleY))[:,:,(2,1,0)]
			img_list.append(np.array(tmp,dtype=np.float32))
		#cv2.imshow("j",tmp)
		#cv2.waitKey(0) & 0xff
		#brak
		for i in range(len(img_list)):
			#print(img_list[i].shape)
			#brak
			img_list[i] /= 255
			img_list[i][:,:,0] -= 0.485#(img_list[i]/127.5) - 1
			img_list[i][:,:,1] -= 0.456
			img_list[i][:,:,2] -= 0.406

			img_list[i][:,:,0] /= 0.229
			img_list[i][:,:,1] /= 0.224
			img_list[i][:,:,2] /= 0.225


		cropped_img_list = fixedCropOnList(img_list, (352, 352))		
		for i in range(len(cropped_img_list)):
			cropped_img_list[i] = torch.from_numpy(cropped_img_list[i].transpose((2, 0, 1)))

		
		return cropped_img_list 

	def __len__(self):
		return len(self.trainList)

class dvsLoader(data.Dataset):

	def __init__(self, imFolderPath, dvsFolderPath,  mode='train'):

		self.trainList, self.dvsList = populateTrainList2(imFolderPath, dvsFolderPath)
		# self.dvsList = populateTrainList2Dvs(dvsFolderPath)
		self.mode = mode
		print("# of training samples:", len(self.trainList))
		print("# of dvs samples:", len(self.dvsList))


	def __getitem__(self, index):

		img_path_list = self.trainList[index]
		dvs_path_list = self.dvsList[index]
		start = random.randint(0,3)
		h,w,c = cv2.imread(img_path_list[0]).shape
		h,w = cv2.imread(dvs_path_list[0], cv2.COLOR_BGR2GRAY).shape

		image = cv2.imread(img_path_list[0])
		dvs = cv2.imread(dvs_path_list[0], cv2.COLOR_BGR2GRAY)

		#print(h,w,c)

		if h > w:
			scaleX = int(360*(h/w))
			scaleY = 360
		elif h <= w:
			scaleX = 360
			scaleY = int(360*(w/h))



		img_list = []
		dvs_list = []

		flip = random.randint(0,1)
		if flip:
			for img_path in img_path_list[start:start+9]:
				tmp = cv2.resize(cv2.imread(img_path), (scaleX,scaleY))[:,:,(2,1,0)]
				img_list.append(np.array(cv2.flip(tmp,1), dtype=np.float32))
			for dvs_path in dvs_path_list[start:start+9]:
				tmp = cv2.resize(cv2.imread(dvs_path, cv2.COLOR_BGR2GRAY), (scaleX,scaleY))
				thresh = 127
				tmp = cv2.threshold(tmp, thresh, 1, cv2.THRESH_BINARY)[1]
				dvs_list.append(np.array(cv2.flip(tmp,1), dtype=np.float32))
		else:
			for img_path in img_path_list[start:start+9]:
				tmp = cv2.resize(cv2.imread(img_path), (scaleX, scaleY))[:,:,(2,1,0)]
				img_list.append(np.array(tmp,dtype=np.float32))
			for dvs_path in dvs_path_list[start:start+9]:
				tmp = cv2.resize(cv2.imread(dvs_path, cv2.COLOR_BGR2GRAY), (scaleX, scaleY))
				thresh = 127
				tmp = cv2.threshold(tmp, thresh, 1, cv2.THRESH_BINARY)[1]
				dvs_list.append(np.array(tmp,dtype=np.float32))
		#cv2.imshow("j",tmp)
		#cv2.waitKey(0) & 0xff
		#brak
		for i in range(len(img_list)):
			#print(img_list[i].shape)
			#brak
			img_list[i] /= 255
			img_list[i][:,:,0] -= 0.485#(img_list[i]/127.5) - 1
			img_list[i][:,:,1] -= 0.456
			img_list[i][:,:,2] -= 0.406

			img_list[i][:,:,0] /= 0.229
			img_list[i][:,:,1] /= 0.224
			img_list[i][:,:,2] /= 0.225


		cropped_img_list, cropped_dvs_list = randomCropOnListDvs(img_list, dvs_list ,(352,352))

		for i in range(len(cropped_img_list)):
			cropped_img_list[i] = torch.from_numpy(cropped_img_list[i].transpose((2, 0, 1)))
		for i in range(len(cropped_dvs_list)):
			cropped_dvs_list[i] = torch.from_numpy(cropped_dvs_list[i])

		
		return cropped_img_list, cropped_dvs_list

	def __len__(self):
		return len(self.trainList)