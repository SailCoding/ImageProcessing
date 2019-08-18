import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('D:\Image\lena512.bmp', 0)
hist, bins = np.histogram(img.flatten(), 256, [0, 256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()

plt.plot(cdf_normalized, color='b')
plt.hist(img.flatten(), 256, [0, 256], color = 'r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.show()

# calculate cdf
cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = 255 * cdf_m / cdf.max()
#cdf_m = (cdf_m - cdf_m.min())*255 / (cdf_m.max() - cdf_m.min())
cdf = np.ma.filled(cdf_m, 0).astype('uint8')

# mapping
img2 = cdf[img]
res = np.hstack((img, img2))
plt.figure(figsize=(10,10))
plt.imshow(res, cmap="gray")
plt.show()


hist, bins = np.histogram(img2.flatten(), 256, [0, 256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()

plt.plot(cdf_normalized, color='b')
plt.hist(img2.flatten(), 256, [0, 256], color = 'r')
plt.xlim([0, 256])
plt.ylim([0, 3000])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.show()

