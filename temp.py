import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from skimage import img_as_ubyte
from skimage.filters import threshold_otsu, threshold_adaptive, gaussian_filter

import cv2

image = cv2.imread("src.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

global_thresh = threshold_otsu(image)
binary_global = image > global_thresh

block_size = 89
binary_adaptive = threshold_adaptive(image, block_size, offset = 10)

cvimg = img_as_ubyte(binary_adaptive)
cvimg = cv2.medianBlur(cvimg, 3)
#cvimg = cv2.GaussianBlur(cvimg, (5, 5), 2)

gs = gridspec.GridSpec(2, 2)
ax0 = plt.subplot(gs[0, 0])
ax1 = plt.subplot(gs[1, 0])
ax2 = plt.subplot(gs[0, 1])
ax3 = plt.subplot(gs[1, 1])
plt.gray()

ax0.imshow(image)
ax0.set_title("Image")

ax1.imshow(binary_global)
ax1.set_title('Global thresholding')

ax2.imshow(binary_adaptive)
ax2.set_title('Adaptive thresholding')

ax3.imshow(cvimg)
ax3.set_title("With blurring")

plt.show()