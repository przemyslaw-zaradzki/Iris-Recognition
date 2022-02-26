##-----------------------------------------------------------------------------
##  Import
##-----------------------------------------------------------------------------
from re import S
import numpy as np
from os import listdir
from fnmatch import filter
import scipy.io as sio
import cupy as cp
from time import time

import warnings
warnings.filterwarnings("ignore")


##-----------------------------------------------------------------------------
##  Function
##-----------------------------------------------------------------------------
def matching(template_extr, mask_extr, temp_dir, threshold=0.38, use_cuda=True):
	"""
	Description:
		Match the extracted template with database.

	Input:
		template_extr	- Extracted template.
		mask_extr		- Extracted mask.
		threshold		- Threshold of distance.
		temp_dir		- Directory contains templates.

	Output:
		List of strings of matched files, 0 if not, -1 if no registered sample.
	"""
	# Get the number of accounts in the database
	n_files = len(filter(listdir(temp_dir), '*.mat'))
	if n_files == 0:
		return -1

	result_list = []
	dir_list = listdir(temp_dir)
	result_list = allmatchingPool(dir_list, template_extr, mask_extr, temp_dir, use_cuda)
	filenames = result_list[0]
	hm_dists = result_list[1]

	# Remove NaN elements
	ind_valid = np.where(hm_dists>0)[0]
	hm_dists = hm_dists[ind_valid]
	filenames = [filenames[idx] for idx in ind_valid]

	# Threshold and give the result ID
	ind_thres = np.where(hm_dists<=threshold)[0]

	# Return
	if len(ind_thres)==0:
		return 0
	else:
		hm_dists = hm_dists[ind_thres]
		filenames = [filenames[idx] for idx in ind_thres]
		ind_sort = np.argsort(hm_dists)
		return [filenames[idx] for idx in ind_sort]


#------------------------------------------------------------------------------
def calHammingDist(template1, mask1, template2, mask2):
	"""
	Description:
		Calculate the Hamming distance between two iris templates.

	Input:
		template1	- The first template.
		mask1		- The first noise mask.
		template2	- The second template.
		mask2		- The second noise mask.

	Output:
		hd			- The Hamming distance as a ratio.
	"""
	# Initialize
	hd = np.nan

	mask = np.logical_or(mask1, mask2)
	nummaskbits = np.sum(mask==1)
	totalbits = template1.size - nummaskbits

	C = np.logical_xor(template1, template2)
	C = np.logical_and(C, np.logical_not(mask))
	bitsdiff = np.sum(C==1)

	if totalbits==0:
		hd = np.nan
	else:
		hd = bitsdiff / totalbits
	
	return hd


#------------------------------------------------------------------------------
def allcalHammingDist(template1, mask1, template2, mask2):
	"""
	Description:
		Calculate the Hamming distance between two iris templates.

	Input:
		template1	- The first template.
		mask1		- The first noise mask.
		template2	- The second template.
		mask2		- The second noise mask.

	Output:
		hd			- The Hamming distance as a ratio.
	"""
	# Initialize
	hd = np.zeros(template2.shape[0])

	allmask1 = np.broadcast_to(mask1, mask2.shape)
	alltemplate1 = np.broadcast_to(template1, template2.shape)
	mask = np.logical_or(allmask1, mask2)
	C = np.logical_xor(alltemplate1, template2)
	C = np.logical_and(C, np.logical_not(mask))
	for i in range(mask2.shape[0]):
		nummaskbits = np.sum(mask[i]==1)
		totalbits = template1.size - nummaskbits
		bitsdiff = np.sum(C[i]==1)
		if totalbits==0:
			hd[i] = np.nan
		else:
			hd[i] = bitsdiff / totalbits

	print("allcalHammingDist")
	return hd


#------------------------------------------------------------------------------
def allcupycalHammingDist(template1, mask1, template2, mask2):
	"""
	Description:
		Calculate the Hamming distance between two iris templates.

	Input:
		template1	- The first template.
		mask1		- The first noise mask.
		template2	- The second template.
		mask2		- The second noise mask.

	Output:
		hd			- The Hamming distance as a ratio.
	"""
	# Initialize
	hd = cp.zeros(template2.shape[0])
	size_vector = template1.size *cp.ones(template2.shape[0])

	allmask1 = cp.broadcast_to(mask1, mask2.shape)
	alltemplate1 = cp.broadcast_to(template1, template2.shape)
	mask = cp.logical_or(allmask1, mask2)
	C = cp.logical_xor(alltemplate1, template2)
	C = cp.logical_and(C, cp.logical_not(mask))

	nummaskbits = cp.sum(mask==1, axis=(1,2))
	bitsdiff = np.sum(C==1, axis=(1,2))
	totalbits = size_vector - nummaskbits
	hd = bitsdiff / totalbits

	# Return
	print("allcupycalHammingDist")
	return hd


#------------------------------------------------------------------------------
def shiftbits(template, noshifts):
	"""
	Description:
		Shift the bit-wise iris patterns.

	Input:
		template	- The template to be shifted.
		noshifts	- The number of shift operators, positive for right
					  direction and negative for left direction.

	Output:
		templatenew	- The shifted template.
	"""
	# Initialize
	templatenew = np.zeros(template.shape)
	width = template.shape[1]
	s = 2 * np.abs(noshifts)
	p = width - s

	# Shift
	if noshifts == 0:
		templatenew = template

	elif noshifts < 0:
		x = np.arange(p)
		templatenew[:, x] = template[:, s + x]
		x = np.arange(p, width)
		templatenew[:, x] = template[:, x - p]

	else:
		x = np.arange(s, width)
		templatenew[:, x] = template[:, x - s]
		x = np.arange(s)
		templatenew[:, x] = template[:, p + x]

	# Return
	return templatenew


#------------------------------------------------------------------------------
def matchingPool(file_temp_name, template_extr, mask_extr, temp_dir):
	"""
	Description:
		Perform matching session within a Pool of parallel computation

	Input:
		file_temp_name	- File name of the examining template
		template_extr	- Extracted template
		mask_extr		- Extracted mask of noise

	Output:
		hm_dist			- Hamming distance
	"""
	# Load each account
	data_template = sio.loadmat('%s%s'% (temp_dir, file_temp_name))
	template = data_template['template']
	mask = data_template['mask']

	# Calculate the Hamming distance
	hm_dist = calHammingDist(template_extr, mask_extr, template, mask)
	return (file_temp_name, hm_dist)


#------------------------------------------------------------------------------
def allmatchingPool(file_temp_name_list, template_extr, mask_extr, temp_dir, use_cuda=True):
	"""
	Description:
		Perform matching session within a Pool of parallel computation

	Input:
		file_temp_name	- File name of the examining template
		template_extr	- Extracted template
		mask_extr		- Extracted mask of noise

	Output:
		hm_dist			- Hamming distance
	"""

	template = np.empty([len(file_temp_name_list), 20, 480])
	mask = np.empty([len(file_temp_name_list), 20, 480])

	for i in range(len(file_temp_name_list)):
		data_template = sio.loadmat('%s%s'% (temp_dir, file_temp_name_list[i]))
		template[i] = data_template['template']
		mask[i] = data_template['mask']

	# Calculate the Hamming distance
	if (use_cuda==True):
		ctemplate_extr = cp.asarray(template_extr)
		cmask_extr = cp.asarray(mask_extr)
		ctemplate = cp.asarray(template)
		cmask = cp.asarray(mask)

		st = time()
		hm_dist = allcupycalHammingDist(ctemplate_extr, cmask_extr, ctemplate, cmask)
		e = time()
		print('\n>>> Verification time: {} [s]\n'.format(e - st))
		hm_dist = cp.asnumpy(hm_dist)

		st = time()
		hm_dist = allcupycalHammingDist(ctemplate_extr, cmask_extr, ctemplate, cmask)
		e = time()
		print('\n>>> Verification time: {} [s]\n'.format(e - st))
		hm_dist = cp.asnumpy(hm_dist)
	else:
		st = time()
		hm_dist = allcalHammingDist(template_extr, mask_extr, template, mask)
		e = time()
		print('\n>>> Verification time: {} [s]\n'.format(e - st))

		st = time()
		hm_dist = allcalHammingDist(template_extr, mask_extr, template, mask)
		e = time()
		print('\n>>> Verification time: {} [s]\n'.format(e - st))

	return (file_temp_name_list, hm_dist)