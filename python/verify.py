##-----------------------------------------------------------------------------
##  Import
##-----------------------------------------------------------------------------
import argparse
from time import time

from fnc.extractFeature import extractFeature
from fnc.matching import matching
import os
import cv2

#------------------------------------------------------------------------------
#	Argument parsing
#------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--file", type=str,
                    help="Path to the file that you want to verify.")

parser.add_argument("--temp_dir", type=str, default="./templates/temp/",
					help="Path to the directory containing templates.")

parser.add_argument("--thres", type=float, default=0.38,
					help="Threshold for matching.")
parser.add_argument("--camera_input", type=str, default="False",
					help="Determine whether to use camera input.")
args = parser.parse_args()


##-----------------------------------------------------------------------------
##  Execution
##-----------------------------------------------------------------------------
# Extract feature
start = time()
print('>>> Start verifying {}\n'.format(args.file))


if(args.camera_input=="True"):
	files_list = os.listdir()
	for file in files_list:
		if("camera_input" in file):
			# Matching
			print(file)
			im = cv2.imread(file)	
			cv2.imshow("Input image", im)
			k = cv2.waitKey(0)
			cv2.destroyAllWindows()
			print("Do you want to continue processing this photo? Y/N")
			answer = input()
			while(answer!="Y" and answer!="N"):
				answer = input()
			if(answer=="Y"):
				print("Processing")
				template, mask, file = extractFeature(file)
				result = matching(template, mask, args.temp_dir, args.thres)
				if result == -1:
					print('>>> No registered sample.')

				elif result == 0:
					print('>>> No sample matched.')
				else:
					print('>>> {} samples matched (descending reliability):'.format(len(result)))
					for res in result:
						print("\t", res)			
else:
	template, mask, file = extractFeature(args.file)


	# Matching
	result = matching(template, mask, args.temp_dir, args.thres)

	if result == -1:
		print('>>> No registered sample.')

	elif result == 0:
		print('>>> No sample matched.')

	else:
		print('>>> {} samples matched (descending reliability):'.format(len(result)))
		for res in result:
			print("\t", res)


# Time measure
end = time()
print('\n>>> Verification time: {} [s]\n'.format(end - start))