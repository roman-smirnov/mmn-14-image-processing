import numpy as np
import matplotlib.pyplot as plt
import cv2

# cosmetics
PIXEL_DENSITY = 175
TITLE_COLOR = 'grey'

# load image
img = cv2.imread('cameraman.jpg', cv2.IMREAD_GRAYSCALE)
print('image size: ', img.shape)

# forward DCT transform
img_f = np.float32(img)
img_dct_f = cv2.dct(img_f)
img_dct = np.uint8(np.abs(img_dct_f))

# forward DFT transform
img_dft_f = np.fft.fft2(img_f)
img_dft = np.uint8(np.log(np.abs(img_dft_f)))

plt.figure(dpi=PIXEL_DENSITY)

# draw original image
plt.subplot(221)
plt.title("Original Image", color=TITLE_COLOR)
plt.imshow(img, cmap="gray")

# draw dct transformed image
plt.subplot(222)
plt.title("DCT Transform", color=TITLE_COLOR)
plt.imshow(img_dct, cmap="gray")

# draw dft transformed image
plt.subplot(223)
plt.title("DFT Transform", color=TITLE_COLOR)
plt.imshow(img_dft, cmap="gray")

# adjust spacing between images
plt.tight_layout()

# display the plot
plt.show()
