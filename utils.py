from cStringIO import StringIO
import numpy as np
import svgwrite
import random
import requests

from scipy.misc import imresize, imsave
from skimage.draw import line_aa

def get_bounds(strokes):
	"""Given a 3-stroke sequence returns the bounds for the respective sketch

	Args:
		strokes: A 3-stroke sequence representing a single sketch

	Returns:
		(min_x, max_x, min_y, max_y): bounds of the sketch
	"""
	min_x, max_x, min_y, max_y = (0, 0, 0, 0)
	abs_x, abs_y = (0, 0)
	
	for i in range(strokes.shape[0]):
		dx, dy = strokes[i, :2]
		abs_x += dx
		abs_y += dy
		min_x = min(min_x, abs_x)
		max_x = max(max_x, abs_x)
		min_y = min(min_y, abs_y)
		max_y = max(max_y, abs_y)

	return (min_x, max_x, min_y, max_y)

def strokes_to_npy(strokes):
	"""Given a 3-stroke sequence returns the sketch in a numpy array

	Args:
		strokes: A 3-stroke sequence representing a single sketch
	
	Returns:
		img: A grayscale 2D numpy array representation of the sketch
	"""
    min_x, max_x, min_y, max_y = get_bounds(strokes)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)
    img = np.zeros(dims, dtype=np.uint8)
    abs_x = 25 - min_x
    abs_y = 25 - min_y
    pen_up = 1
    for i in range(strokes.shape[0]):
        dx, dy = strokes[i, :2]
        
		if pen_up == 0:
            rr, cc, val = line_aa(abs_x, abs_y, abs_x + dx, abs_y + dy)
            img[rr, cc] = val * 255
        
		abs_x += dx
        abs_y += dy
        pen_up = strokes[i, 2]
	
	# TODO: Why do we need to transpose? Fix get_bounds accordingly
    return img.T

def reshape_to_square(img, size=256):
	"""Given any size numpy array return

	Args:
		img: A grayscale 2D numpy array representation of the sketch
	
	Returns:
		img_sq: A grayscale 2D numpy array representation of the sketch fitted
				in a size x size square.
	"""
	# TODO: make sure this formula is correct
    img_resize = imresize(img, float(size) / max(img.shape))
    w_, h_ = img_resize.shape
    x, y = ((size - w_) / 2, (size - h_) / 2)
    
	img_sq = np.zeros((size, size), dtype=np.uint8)
    img_sq[x:x + w_, y:y + h_] = img_resize
    
	return img_sq

class DataLoader(object):

	def __init__():
		
	
