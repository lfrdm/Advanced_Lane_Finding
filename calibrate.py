import numpy as np
import cv2
import matplotlib.pyplot as plt	
import matplotlib.image as mpimg
import glob
import pickle

# function to undistort image
def cal_undistort(shape, objpoints, imgpoints):
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
    
    #undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    return mtx, dist

images = glob.glob('camera_cal/calibration*.jpg')

# set object and imgpoints
objpoints = []
imgpoints = []

# iterate over all images
for fname in images:
	# read img
	img = mpimg.imread(fname)
	print(fname)

	# convert imgae to grayscale
	gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

	# find chessboard corners
	ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
	six = True

	if ret == False:
		ret, corners = cv2.findChessboardCorners(gray, (9,5), None)
		six = False

	# create grid points of chessboard 9x6 or 9x5
	if six == True:
		objp = np.zeros((6*9,3), np.float32)
		objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
	else: 
		objp = np.zeros((5*9,3), np.float32)
		objp[:,:2] = np.mgrid[0:9,0:5].T.reshape(-1,2)

	print(ret)
	# if corners are found, add object points 
	if ret == True:
		imgpoints.append(corners)
		objpoints.append(objp)

mtx, dist = cal_undistort(gray.shape[::-1], objpoints, imgpoints)

img = plt.imread('camera_cal/calibration1.jpg')

undist = cv2.undistort(img, mtx, dist, None, mtx)
#plt.imshow(undist)
#plt.show()
plt.imsave('calibrated1.jpg',undist)

# Saving mtx and dist:
f = open('cal.pckl', 'wb')
pickle.dump([mtx, dist], f)
f.close()



