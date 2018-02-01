import numpy as np
import cv2
from matplotlib import pyplot as plt

# cosmetics
PLOT_PIXEL_DENSITY = 175
POINT_SIZE = 15
POINT_COLOR = 'red'
TITLE_COLOR = 'grey'

# init plot with better pixel density
fig = plt.figure(dpi=PLOT_PIXEL_DENSITY)

# use this to get a console printout of subplot click pixel coordinates
# fig.canvas.mpl_connect('button_press_event', lambda event: print(round(event.xdata), round(event.ydata)))

# reference points
src_pts = np.array([[0, 0], [255, 0], [0, 255], [255, 255]])
src_x, src_y = src_pts.T
dst_pts = np.array([[80, 30], [405, 30], [85, 300], [345, 292]])
dst_x, dst_y = dst_pts.T

# read the images from disk as greyscale
src_im = cv2.imread('cameraman.jpg', cv2.IMREAD_GRAYSCALE)  # source image
dst_im = cv2.imread('cameraman2.jpg', cv2.IMREAD_GRAYSCALE)  # destination image

# Calculate estimated homography matrix
hom, status = cv2.findHomography(src_pts, dst_pts)

# generate the registered image
reg_im = cv2.warpPerspective(src_im, hom, (dst_im.shape[1], dst_im.shape[0]))

# draw original image
plt.subplot(221)
plt.title("Original Image", color=TITLE_COLOR)
plt.imshow(src_im, cmap="gray")
plt.scatter(src_x, src_y, c=POINT_COLOR, s=POINT_SIZE)  # draw the corresponding points

# draw transformed image
plt.subplot(222)
plt.title("Transformed Image", color=TITLE_COLOR)
plt.imshow(dst_im, cmap="gray")
plt.scatter(dst_x, dst_y, c=POINT_COLOR, s=POINT_SIZE)  # draw the corresponding points

# draw image produced from homography estimate
plt.subplot(223)
plt.title("Estimate Transformed Image", color=TITLE_COLOR)
plt.imshow(reg_im, cmap="gray")

# draw the homography matrix as text
plt.subplot(224)
plt.title("Homography Matrix", color=TITLE_COLOR)
plt.text(0.5, 0.5, np.array2string(hom.astype(dtype=int)), ha='center', va='center')
# hide text subplot labels and ticks
plt.tick_params(axis='both', which='both', bottom='off', labelbottom='off', left='off', labelleft='off')

# adjust spacing between images
plt.tight_layout()

# display the plot
plt.show()
