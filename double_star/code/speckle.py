from astropy.io import fits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
#Рассчет среднего кадра 
data = fits.open('speckledata.fits')[2].data
T, H, W = data.shape
xc = W//2
yc = H//2
first_image = data[0]
plt.imshow(first_image)
mean_image = np.mean(data, axis = 0)
plt.imshow(mean_image, cmap='grey')
plt.savefig('mean.png')
#Переход в Фурье-пространство
data = data.astype(float)
fourier  = np.fft.fft2(data)
power  = np.abs(fourier)**2
mean_power = np.mean(power, axis= 0)
img = np.fft.fftshift(mean_power)
vmin = np.percentile(img, 10.0)      
vmax = np.percentile(img, 99.0)   
plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
plt.savefig('fourier.png')
#усреднение по углам
radius  = 50
Y, X = np.ogrid[:H, :W]
R = np.sqrt((Y-yc)**2 + (X - xc)**2)
mask_inside = R <= radius
img_spy = np.ma.masked_array(img, mask= mask_inside)
mean_img_trash  = np.mean(img_spy)
img_without_trash = img - mean_img_trash
plt.imshow(img_without_trash, cmap='gray', vmin=vmin, vmax=vmax)
angles = np.arange(0, 180, 5)
rotate_images = []
for i in range(0, 180, 5):
    img_angle = rotate(img_without_trash, i, reshape = False)
    rotate_images.append(img_angle)
rotate_images = np.array(rotate_images)
mean_img_for_angles = np.mean(rotate_images, axis=0)
plt.imshow(mean_img_for_angles, cmap='gray', vmin = np.percentile(mean_img_for_angles, 8.0), vmax = np.percentile(mean_img_for_angles, 93.0))
plt.savefig('rotaver.png')
#Отсечение частот
mask_outside = R >= radius
plt.imshow(mask_outside)
img_norm = img_without_trash/(mean_img_for_angles+1e-12)
img_norm[~np.isfinite(img_norm)] = 0.0
#plt.imshow(img_norm, cmap='grey', vmin= v_testmin, vmax=v_testmax)
img_frequency = np.ma.masked_array(img_norm, mask= mask_outside)
img_norm_filled = img_frequency.filled(0)
plt.imshow(img_norm_filled , cmap='gray')
v_testmin = np.percentile(img_norm_filled, 5.0)
v_testmax = np.percentile(img_norm_filled, 99.0)
plt.imshow(img_norm_filled , cmap='gray', vmin= v_testmin, vmax=v_testmax)

img_norm_filled1 = np.fft.ifftshift(img_norm_filled)
acf = np.fft.ifft2(img_norm_filled1).real
vmin1 = np.percentile(acf, 5.0)      
vmax1 = np.percentile(acf, 98.0)  
plt.imshow(acf, cmap='gray', vmin=vmin1, vmax=vmax1 )
plt.savefig('binary.png')