import cv2
import pickle
import matplotlib.pyplot as plt	
import numpy as np
import scipy.misc
from moviepy.editor import VideoFileClip

# define as globals to be used in run_image
#global mtx, dist, detected, left_lane, right_lane

def abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100):
	"""
	Takes an image, gradient orientation, and threshold min/max values
	"""
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# Apply x or y gradient with the OpenCV Sobel() function
	# and take the absolute value
	if orient == 'x':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
	if orient == 'y':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
	# Rescale back to 8 bit integer
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
	# Create a copy and apply the threshold
	binary_output = np.zeros_like(scaled_sobel)
	# Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
	binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

	# Return the result
	return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 100)):
	"""
	Return the magnitude of the gradient
	for a given sobel kernel size and threshold values
	"""
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# Take both Sobel x and y gradients
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Calculate the gradient magnitude
	gradmag = np.sqrt(sobelx**2 + sobely**2)
	# Rescale to 8 bit
	scale_factor = np.max(gradmag)/255
	gradmag = (gradmag/scale_factor).astype(np.uint8)
	# Create a binary image of ones where threshold is met, zeros otherwise
	binary_output = np.zeros_like(gradmag)
	binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

	# Return the binary image
	return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
	"""
	Return the direction of the gradient
	for a given sobel kernel size and threshold values
	"""
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# Calculate the x and y gradients
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Take the absolute value of the gradient direction,
	# apply a threshold, and create a binary image result
	absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
	binary_output =  np.zeros_like(absgraddir)
	binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

	# Return the binary image
	return binary_output


def hls_thresh(img, thresh=(100, 255)):
	"""
	Convert RGB to HLS and threshold to binary image using S channel
	"""
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	s_channel = hls[:,:,2]
	binary_output = np.zeros_like(s_channel)
	binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
	return binary_output


def combined_thresh(img):
	abs_bin = abs_sobel_thresh(img, orient='x', thresh_min=55, thresh_max=255)
	mag_bin = mag_thresh(img, sobel_kernel=5, mag_thresh=(55, 255))
	dir_bin = dir_threshold(img, sobel_kernel=11, thresh=(0.7, 1.3))
	hls_bin = hls_thresh(img, thresh=(175, 255))

	combined = np.zeros_like(dir_bin)
	combined[(abs_bin == 1 | ((mag_bin == 1) & (dir_bin == 1))) | hls_bin == 1] = 1

	return combined

#Perform perspective transform
def warpPerspective(img):
 
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([
    	[705, 460],
    	[1035, 670],
        [286, 670],
        [582, 460]])
    dst = np.float32([
    	[880, 0], 
    	[880, 720], 
        [400, 720],
        [400, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size)

    return warped, M_inv

def find_lines(binary_warped):

	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
	# Create an output image to draw on and  visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(binary_warped.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 25
	# Set minimum number of pixels found to recenter window
	minpix = 250
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
	    # Identify window boundaries in x and y (and right and left)
	    win_y_low = binary_warped.shape[0] - (window+1)*window_height
	    win_y_high = binary_warped.shape[0] - window*window_height
	    win_xleft_low = leftx_current - margin
	    win_xleft_high = leftx_current + margin
	    win_xright_low = rightx_current - margin
	    win_xright_high = rightx_current + margin
	    # Draw the windows on the visualization image
	    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
	    (0,255,0), 2) 
	    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
	    (0,255,0), 2) 
	    # Identify the nonzero pixels in x and y within the window
	    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
	    (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
	    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
	    (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
	    # Append these indices to the lists
	    left_lane_inds.append(good_left_inds)
	    right_lane_inds.append(good_right_inds)

	    # If you found > minpix pixels, recenter next window on their mean position
	    if len(good_left_inds) > minpix:
	        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
	    if len(good_right_inds) > minpix:        
	        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extrt left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	return left_fit, right_fit, left_lane_inds, right_lane_inds

def fast_find_lines(binary_warped, left_fit, right_fit):
	# Assume you now have a new warped binary image 
	# from the next frame of video (also called "binary_warped")
	# It's now much easier to find line pixels!
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	margin = 25
	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
	left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
	left_fit[1]*nonzeroy + left_fit[2] + margin))) 

	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
	right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
	right_fit[1]*nonzeroy + right_fit[2] + margin)))  

	# Again, extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]
	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	return left_fit, right_fit, left_lane_inds, right_lane_inds

def curvature(binary_warped, left_lane_inds, right_lane_inds):

	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	# Extrt left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 

	# image height -1
	y_eval = 719

	# pixel to world coordinates
	# ca. 30m per 720px and 3.7 width of 480px
	ym_per_pix = 30/720 # meters per pixel in y dimension
	xm_per_pix = 3.7/480 # meters per pixel in x dimension

	# fit to new points
	left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

	# calc curvature
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
	
	return left_curverad, right_curverad

def offset(undist, left_fit, right_fit):
	
	# image height -1
	y_lowest = 719
	x_lowest_left = left_fit[0]*y_lowest**2 + left_fit[1]*y_lowest + left_fit[2]
	x_lowest_right = right_fit[0]*y_lowest**2 + right_fit[1]*y_lowest + right_fit[2]
	offset = abs(undist.shape[1]/2 - (x_lowest_left + x_lowest_right)/2)

	# pixel to world coordinates
	xm_per_pix = 3.7/480 # meters per pixel in x dimension

	return offset*xm_per_pix

def lane_distance(left_fit, right_fit):
	
	# image height -1
	y_lowest = 719
	x_lowest_left = left_fit[0]*y_lowest**2 + left_fit[1]*y_lowest + left_fit[2]
	x_lowest_right = right_fit[0]*y_lowest**2 + right_fit[1]*y_lowest + right_fit[2]
	offset = x_lowest_right - x_lowest_left

	return offset

def draw(warped, Minv,left_fit, right_fit, offset, left_curvature, right_curvature):

	frame_cnt = 0
	
	# y values
	ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
	
	# x values from fits
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# color warp
	color_warp = np.zeros((720, 1280, 3), dtype='uint8')

	# left and right points of lanes
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# fill area
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# warp back on undist image
	newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0])) 
	
	# combine images
	result = cv2.addWeighted(warped, 1, newwarp, 0.3, 0)

	# write curvature every 15 frames
	if(frame_cnt%15==0):
		meanCurvature = (left_curvature + right_curvature)/2
		string = 'Curvature: %.1f m' % meanCurvature
		result = cv2.putText(result, string, (30,40), 0, 1, (0,0,0), 2, cv2.LINE_AA)

	# write offset every frame
	string = 'Offset: %.1f m' % offset
	result = cv2.putText(result, string, (30,70), 0, 1, (0,0,0), 2, cv2.LINE_AA)

	frame_cnt += 1

	return result

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # list of last ploynomial coefficient f(x) = a*x^2+by+c
        self.a = []
        self.b = []
        self.c = []
        self.curvature = []

    def addCoefficients(self, best_fit, curvature):
    	# adds coefficients to lists with maximum length of 5
    	self.a.append(best_fit[0])
    	self.b.append(best_fit[1])
    	self.c.append(best_fit[2])
    	self.curvature.append(curvature)

    	if(len(self.a)>5):
    		self.a.pop(0)
    		self.b.pop(0)
    		self.c.pop(0)

    	if(len(self.curvature)>10):
    		self.curvature.pop(0)

    def getAverage(self):
    	# get the average of the coefficients lists
    	return (np.mean(self.a), np.mean(self.b), np.mean(self.c)), np.mean(self.curvature)

# Loading mtx and dist:
f = open('cal.pckl', 'rb')
mtx, dist = pickle.load(f)
f.close()

# global variables
detected = False
left_lane = Line()
right_lane = Line()

def run_image(img):

	global detected, left_lane, right_lane, dist, mtx

	# undistort image
	undist = cv2.undistort(img, mtx, dist, None, mtx)

	# edges
	edges = combined_thresh(undist)

	# warp image and mask
	warped, M_inv = warpPerspective(edges)
	warped[:,980:] = 0
	warped[:,:300] = 0

	if not detected:

		# perform line fit
		left_fit, right_fit, left_lane_inds, right_lane_inds = find_lines(warped)
		# calc curvature
		left_curvature, right_curvature = curvature(warped, left_lane_inds, right_lane_inds)

		detected = True
		
		left_lane.addCoefficients(left_fit, left_curvature)
		right_lane.addCoefficients(right_fit, right_curvature)

	else:

		left_fit, _ = left_lane.getAverage()
		right_fit, _ = right_lane.getAverage()
		# perform line fit
		left_fit, right_fit, left_lane_inds, right_lane_inds = fast_find_lines(warped, left_fit, right_fit)

		if (len(left_fit)>0 and len(right_fit)>0):

			# calc curvature
			left_curvature, right_curvature = curvature(warped, left_lane_inds, right_lane_inds)

			detected = True

			# add to lane
			left_lane.addCoefficients(left_fit, left_curvature)
			right_lane.addCoefficients(right_fit, right_curvature)
		else:
			detected = False

	# get
	left_fit, left_curvature = left_lane.getAverage()
	right_fit, right_curvature = right_lane.getAverage()

	# calculate offset from center of road
	road_offset = offset(undist, left_fit, right_fit)

	# draw on image
	result = draw(undist, M_inv, left_fit, right_fit, road_offset, left_curvature, right_curvature)

	return result

def run_video(input, output):
	# run video
	video = VideoFileClip(input)
	out_video = video.fl_image(run_image)
	out_video.write_videofile(output, audio=False)

if __name__ == '__main__':
	
	run_video('project_video.mp4', 'out_1.mp4')





