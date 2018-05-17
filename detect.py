import numpy as np
from scipy.spatial import distance
import cv2 as cv    
import sys 

def find_area(point):
	l = abs(point[0][0] - point[0][1])
	w = abs(point[1][0] - point[1][1])
	print (l,w)
	return l*w

# def is_similar(point1, point2):
# 	first_point = abs(point1[0][0] - point2[0][0])
# 	second_point = abs(point1[1][1] - point2[1][1])
# 	if first_point == second_point:
# 		return True
# 	if ( first_point < point_threshold and not second_point < point_threshold):
# 		return True
# 	elif ( second_point < point_threshold and not first_point < point_threshold):
# 		return True
# 	elif (first_point < point_threshold and second_point < point_threshold):
# 		return True
# 	return False

point_threshold = 10

real_img = cv.imread('inv1.png', -1)

img = cv.cvtColor(real_img, cv.COLOR_BGR2GRAY)

rows = img.shape[0]
cols = img.shape[1]

vis = img[0:int(rows)][:].copy()

mser = cv.MSER_create()
regions, boxes = mser.detectRegions(vis)
hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions]

points = []
mid_points = []

for i,contour in enumerate(hulls):
	x,y,w,h = cv.boundingRect(contour)
	point = [(y,y+h),(x,x+w)]
	mid_point = (y+(h/2),x+(w/2))
	print(point, mid_point)
	if i == 0:
		points.append(point)
		mid_points.append(mid_point)
		continue
	else:
		print ("mps",mid_points)
		dist= distance.euclidean(mid_point, mid_points[-1])
		print ("distance",dist)
		if dist > point_threshold:
			# New word
			mid_points.append(mid_point)
			points.append(point)
			# cv.imshow('img', real_img[point[0][0]:point[0][1],point[1][0]:point[1][1]])
			# cv.waitKey(0)
			# cv.destroyAllWindows()
		else:
			# Similar point
			Area1 = find_area(point)
			Area2 = find_area(points[-1])
			print(Area1,">",Area2)
			if Area1 > Area2:
				# If this point has an area greater than the previous similar point
				points[-1] = point
				mid_points[-1] = mid_point
				# cv.imshow('img', real_img[point[0][0]:point[0][1],point[1][0]:point[1][1]])
				cv.waitKey(0)
				cv.destroyAllWindows()
				continue
			input()

# for point in points:
# 	cv.imshow('img', real_img[point[0][0]:point[0][1],point[1][0]:point[1][1]])
	# cv.waitKey(0)
	# cv.destroyAllWindows()
