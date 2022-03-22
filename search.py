#
# Check for PCB aasembly defects e.g missing components, solder joints, component tilting etc
#
from skimage.metrics import structural_similarity as compare_ssim
import argparse
import imutils
import cv2
import numpy as np
import copy

# font specs
font = cv2.FONT_HERSHEY_SIMPLEX
# fontScale
fontScale = 0.3
# Line thickness of 1 px
thickness = 1

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--o", required=True,
                help="Original image")
ap.add_argument("-s", "--d", required=True,
                help="Test Image")
args = vars(ap.parse_args())

# load the two input images
imageA = cv2.imread(args["o"])
imageB = cv2.imread(args["d"])

#dups
DimageA = copy.copy(imageA)

# convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255,
                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

i = 0
cv2.putText(imageA, "ORIGINAL", (4, 10), font,
                fontScale, (0, 255, 255), thickness, cv2.LINE_AA)
cv2.putText(imageB, "TEST", (4, 10), font,
                fontScale, (255, 255, 0), thickness, cv2.LINE_AA)
# loop over the contours
for c in cnts:
    # compute the bounding box of the contour and then draw the
    # bounding box on both input images to represent where the two
    # images differ
    
    (x, y, w, h) = cv2.boundingRect(c)
    i += 1
    cv2.putText(imageA, "{}-{},{}".format(i, x, y), (x-2, y-6), font,
                fontScale, (0, 255, 255), thickness, cv2.LINE_AA)
    cv2.rectangle(imageA, (x-2, y-2), (x + w, y + h), (255, 0, 0), 2)

    cv2.putText(imageB, "{}-{},{}".format(i, x, y), (x-2, y-6), font,
                fontScale, (255, 255, 0), thickness, cv2.LINE_AA)
    cv2.rectangle(imageB, (x-2, y-2), (x + w, y + h), (0, 0, 255), 2)


# show the output images
#cv2.imshow("Original with no markers", DimageA)
cv2.imshow("Test PCB Board - Defects/Missing Parts: {}".format(i), np.concatenate((imageA, imageB), axis=1))
#cv2.imshow("Test PCB Board - Defects/Missing Parts: {}".format(i), imageB)
cv2.imshow("Diff", diff)
cv2.imshow("Thresh", thresh)
key = cv2.waitKey(0)

cv2.destroyAllWindows()

# python isearch.py --o images/og1.png --d images/og2.png
# tesseract
# scikit
# tensorflow
