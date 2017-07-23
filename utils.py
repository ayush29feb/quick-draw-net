from cStringIO import StringIO
import logging
import numpy as np
import os
import random
import requests
import svgwrite
import urllib

from scipy.misc import imresize
from skimage.draw import line_aa

QUICK_DRAW_BASE_URL = 'https://storage.googleapis.com/quickdraw_dataset/sketchrnn'
QUICK_DRAW_CATEGORIES_URL = 'https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt'

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
	"""Class for loading data."""

	def __init__(self,
				batch_size=25,
				img_size=256,
				dataset_base_url=QUICK_DRAW_BASE_URL,
				categories_file=QUICK_DRAW_CATEGORIES_URL,
				data_format='bitmap',
				datastore_dir='/tmp/quick-draw-net/data/',
				shuffle=False):
		self.batch_size = batch_size
		self.img_size = img_size
		self.data_format = data_format
		self.datastore_dir = datastore_dir
		self.dataset_base_url = dataset_base_url

		self.load_categories_url(categories_file)

	def load_categories_url(self, categories_file):
		"""Loads the categories line seperated file with all the categories to
		loaded by the DataLoader. Adds the path to the dataset file in the
		categories_path dictionary.

		Args:
			categories_file: path to the line seperated file with desired category names
		"""
		self.categories_path = {}
		self.categories_url = {}
		categories = []
		if categories_file.startswith('http://') or categories_file.startswith('https://'):
			response = requests.get(categories_file)
			if response.status_code != 200:
				raise IOError(msg='Request to %s responded with status %d'
						% (categories_file, response.status_code))
			categories = response.content.split('\n')
		else:
			categories = open(categories_file).read().split('\n')

		# filter skiped lines and categories that are skipped (starts with #)
		categories = list(filter(lambda x: len(x) > 0 and x[0] != '#', categories))
		# add the urls if they are valid and give status 200

		for category in categories:
			category_url = os.path.join(self.datastore_dir, '%s.npz' % category)
			if os.path.isfile(category_url):
				self.categories_path[category] = category_url

			category_url = os.path.join(self.dataset_base_url, '%s.npz' % category)
			response = requests.head(category_url)
			if response.status_code == 200:
				self.categories_url[category] = category_url
			else:
				logging.warn('Category %s was not found' % category)

			category_full_url = os.path.join(self.datastore_dir, '%s.full.npz' % category)
			if os.path.isfile(category_full_url):
				self.categories_path[category] = category_full_url

			category_full_url = os.path.join(self.dataset_base_url, '%s.full.npz' % category)
			response = requests.head(category_full_url)
			if response.status_code == 200:
				self.categories_url['%s.full' % category] = category_full_url
			else:
				logging.warn('Category %s (full) was not found' % category)

	def download_dataset(self, category, full=False):
		"""Downloads the dataset for the requested category to the data_dir

		Args:
			category: category of the dataset to be downloaded
			full: a boolean flag to download the complete dataset file for
					the given category
		"""
		category_ = category if not full else '%s.full' % category

		if category_ not in self.categories_path and category_ not in self.categories_url:
			raise ValueError(msg='Could not download category %s because if does not exist' % category_)

		# download the data if don't have a local copy
		if category_ not in self.categories_path:
			response = requests.get(self.categories_url[category_])
			if response.status_code != 200:
				raise IOError(msg='Request to %s responded with status %d'
						% (self.categories_url[category_], response.status_code))

			with open(os.path.join(self.datastore_dir, '%s.npz' % category_), 'w') as handle:
				for block in response.iter_content(1024):
					handle.write(block)

			self.categories_path[category_] = os.path.join(self.datastore_dir, '%s.npz' % category_)

		# load the category data into memory
		strokes = np.load(StringIO(open(self.categories_path[category_]).read()))

		return strokes['train']

	def next_batch(self):
		pass