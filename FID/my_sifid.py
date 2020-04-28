# example of calculating the frechet inception distance in Keras
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize

from PIL import Image
import numpy as np
import os

GAN_DIR = "C:/Users/samtu/Desktop/4995_DL/cSinGAN/FID/sinGANimgs"
IMG_NAMES = ["balloons", "birds", "zebra", "colusseum", "starry_night"]
SUFFIX = "png"


# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)


# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2) ** 2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid


# 1 on 1 comparison between pairs of real images and fake images
def main():
	# prepare the inception v3 model
	model = InceptionV3(include_top=False, pooling='avg', input_shape=(300, 300, 3))
	
	fid_results = []
	
	print("\n")
	
	for IMG_NAME in IMG_NAMES:
		print("Calculating image: " + IMG_NAME)
		# load images
		img_real = np.array(Image.open(GAN_DIR + "/real_img/" + IMG_NAME + "." + SUFFIX))
		img_fake = np.array(Image.open(GAN_DIR + "/fake_img/" + IMG_NAME + "." + SUFFIX))
		print(IMG_NAME + ' images loaded')
		
		# resize images
		img_real = scale_images(img_real, (300, 300, 3))
		img_fake = scale_images(img_fake, (300, 300, 3))
		print("Images scaled to desired dimensions")
		
		# pre-process images
		img_real = preprocess_input(img_real)
		img_fake = preprocess_input(img_fake)
		print("Images pre-processed")
		
		# fid between real and fake images
		fid = calculate_fid(model, img_real, img_fake)
		print('FID of ' + IMG_NAME + ': %.3f' % fid)
		
		fid_results += [round(fid, 5)]
		
		print("\n")
	
	print ("fid results: ", fid_results, "\n")
	
	print ("fid average: ", sum(fid_results) / len(fid_results), "\n")


# 1 on many comparison between pairs of real images and fake images
def main1():
	# prepare the inception v3 model
	model = InceptionV3(include_top=False, pooling='avg', input_shape=(300, 300, 3))
	
	fid_raw_results = []
	
	fid_avg_results = []
	
	print("\n")
	
	for IMG_NAME in IMG_NAMES:
		current_fid_result = []
		
		print("Calculating image: " + IMG_NAME + "\n")
		# load real image
		img_real = np.array(Image.open(GAN_DIR + "/" + IMG_NAME + "." + SUFFIX))
		print("Real image loaded")
		
		# resize real image
		img_real = scale_images(img_real, (300, 300, 3))
		print("Real image resized")
		
		# pre-process real image
		img_real = preprocess_input(img_real)
		print("Real image pre-processed")
		
		print("Loading fake " + IMG_NAME + " images")
		
		fake_imgs = os.listdir(GAN_DIR + "/" + IMG_NAME)
		fake_imgs.sort()
		
		for filename in fake_imgs:
			print("Loading image ", filename, " of ", len(fake_imgs), "fake ", IMG_NAME, " images")
			# load fake image
			img_fake = np.array(Image.open(GAN_DIR + "/" + IMG_NAME + "/" + filename))
			print(filename + ' fake image loaded')
			
			# resize fake image
			img_fake = scale_images(img_fake, (300, 300, 3))
			print("Fake image" + filename + " scaled to desired dimensions")
			
			# pre-process images
			img_fake = preprocess_input(img_fake)
			print("Fake image " + filename + " pre-processed")
			
			# fid between real and fake images
			fid = calculate_fid(model, img_real, img_fake)
			print("FID between real and fake image ", filename, ": ", fid)
			
			current_fid_result += [round(fid, 5)]
			
			print("\n")
		
		print("FID result for ", IMG_NAME, current_fid_result, "\n")
		fid_raw_results += [current_fid_result]
		
		avg_fid = round(sum(current_fid_result) / len(current_fid_result), 5)
		print("Averaged FID result for ", IMG_NAME, avg_fid)
		fid_avg_results += [avg_fid]
		
		print("\n")
	
	print("fid raw results: ", fid_raw_results, "\n")
	
	print("fid averages: ", fid_avg_results, "\n")
	
	print("on images: ", IMG_NAMES)


main1()
