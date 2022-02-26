##-----------------------------------------------------------------------------
##  Import
##-----------------------------------------------------------------------------
import argparse
from time import time

from fnc.cuda_extractFeature import extractFeature
from fnc.cuda_matching import matching


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
parser.add_argument("--use_cuda", type=str, default="True",
					help="Determine whether to use CUDA.")
args = parser.parse_args()


##-----------------------------------------------------------------------------
##  Execution
##-----------------------------------------------------------------------------
# Extract feature

print('>>> Start verifying {}\n'.format(args.file))
template, mask, file = extractFeature(args.file)

start = time()
# Matching
result = matching(template, mask, args.temp_dir, args.thres, args.use_cuda=="True")

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