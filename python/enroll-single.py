##-----------------------------------------------------------------------------
##  Import
##-----------------------------------------------------------------------------
import argparse, os
from time import time
from scipy.io import savemat

from fnc.extractFeature import extractFeature
import cv2

#------------------------------------------------------------------------------
#	Argument parsing
#------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--file", type=str,
                    help="Path to the file that you want to verify.")

parser.add_argument("--temp_dir", type=str, default="./templates/temp/",
					help="Path to the directory containing templates.")
parser.add_argument("--camera_input", type=str, default="False",
					help="Determine whether to use camera input.")
args = parser.parse_args()


##-----------------------------------------------------------------------------
##  Execution
##-----------------------------------------------------------------------------
start = time()
#args.file = "../CASIA1/001_1_1.jpg"

if(args.camera_input=="True"):
    files_list = os.listdir()
    for file in files_list:
        if("camera_input" in file):
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
                # Extract feature
                print('>>> Enroll for the file ', file)
                template, mask, file = extractFeature(file)

                print("Do you want to save the code for this image of the iris? Y/N")
                answer = input()
                while(answer!="Y" and answer!="N"):
                    answer = input()
                if(answer=="Y"):
                    # Save extracted feature
                    basename = os.path.basename(file)
                    out_file = os.path.join(args.temp_dir, "%s.mat" % (basename))
                    savemat(out_file, mdict={'template':template, 'mask':mask})
                    print('>>> Template is saved in %s' % (out_file))
else:
    # Extract feature
    print('>>> Enroll for the file ', args.file)
    template, mask, file = extractFeature(args.file)

    print("Do you want to save the code for this image of the iris? Y/N")
    answer = input()
    while(answer!="Y" and answer!="N"):
        answer = input()
    if(answer=="Y"):
        # Save extracted feature
        basename = os.path.basename(file)
        out_file = os.path.join(args.temp_dir, "%s.mat" % (basename))
        savemat(out_file, mdict={'template':template, 'mask':mask})
        print('>>> Template is saved in %s' % (out_file))

end = time()
print('>>> Enrollment time: {} [s]\n'.format(end-start))