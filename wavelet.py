# Python code to read image
import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("wave.jpg", cv2.IMREAD_GRAYSCALE)

coeffs = pywt.dwt2(img, 'db7')

cA, (cH, cV, cD) = coeffs
print(np.shape(cA))
new_cD = np.zeros([372, 493])
coeffs=cA,(cH,cV,new_cD)
new_img = pywt.idwt2(coeffs, 'db7')
print(np.shape(cD))


dif=np.subtract(img,new_img)
print(dif)
plt.subplot(2,2,1)
plt.imshow(np.uint8(cA),cmap='gray')
plt.show
plt.subplot(2,2,2)
plt.imshow(np.uint8(cH),cmap='gray')
plt.show
plt.subplot(2,2,3)
plt.imshow(np.uint8(cV),cmap='gray')
plt.subplot(2,2,4)
plt.imshow(np.uint8(cD),cmap='gray')
plt.show()
plt.imshow(new_img,cmap='gray')
plt.show()
plt.imshow(img,cmap='gray')
plt.show()
plt.imshow(dif,cmap='gray')
plt.show()
