# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 14:16:05 2021

@author: imrul
"""

import numpy as np
import matplotlib.pyplot as plt

import scipy
import scipy.signal
from scipy.ndimage import gaussian_filter1d
import math
# from analyse_edge_image import deduce_bayer_pattern
# from analyse_distortion import find_edge_orientation, find_edge, reduce_1d
# from analyse_edge_image import locate_edge, reorient_image
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
from scipy.fft import fft, ifft,fftshift
from ROI_manager_user import userROI
#%%
def check_b2w_transition(raw_image,channel=1):
    row,col,ch=np.shape(raw_image)
    block_size=5
    left_block=raw_image[row//2-block_size:row//2+block_size,col//4-block_size:col//4+block_size,channel] # get a small crop at the left side
    right_block=raw_image[row//2-block_size:row//2+block_size,3*col//4-block_size:3*col//4+block_size,channel]
    tol=10
    raw_image_f=np.copy(raw_image)
    if np.mean(left_block)>np.mean(right_block)+tol:
        print('Warning: Algorithm demands black to white transition. Applying Horizontal flip')
        raw_image_f=np.fliplr(raw_image)
        
    return raw_image_f
        
    
    
    

def extract_aligned_edge(raw_image, width, channel=1, force_tx=None, thresholds=(0.25,0.75)):
    """Extract x, I coordinates for each row of the image, aligned to the edge
    
    raw_image should be an NxMx3 RGB image, with the dark-bright transition along the middle axis
    width will be the width of the output rows (should be less than the width of the raw image)
        NB this width is relative to the original image - as each row will only contain the 
        pixels with the selected colour, the rows will have half this many points.
    channel is the channel of the image to extract (0-2)
    thresholds specifies the region of I (relative to min/max)
    
    Return Value: [(x, I, tx)]
    x and I are cropped 1D arrays, each representing the x and I coordinates for one row.  The range of x
    will be approximately -widdth/2 to width/2, though there will usually be width/2 points.  tx is the 
    x coordinate where the transition was detected (i.e. the shift between the x coordinates and the pixels
    of the original image).
    """
    # bayer = deduce_bayer_pattern(raw_image)[:,:,channel]
    aligned_rows = []
    for i in range(raw_image.shape[0]):
        # row_bayer = bayer[i % bayer.shape[0], :]
        # print(i)
        # if np.any(row_bayer): # if there are active pixels of this channel in this row
            # x = np.argmax(row_bayer) # find the position of the first relevant pixel in the row
            # active_pixel_slice = slice(x, None, bayer.shape[1]) 
                # This slice object takes every other pixel, starting at x (x==0 or 1)

            # Extract the position and intensity of all of the pixels that are active
            # x = np.arange(raw_image.shape[1])[active_pixel_slice]
            # I = raw_image[i,active_pixel_slice,channel]
        x = np.arange(raw_image.shape[1])
        I = raw_image[i,:,channel]
        
        
        

        # Crop out the part where it goes from black to white (from 25% to 75% intensity)
        normI = (I - np.min(I) + 0.0)/(np.max(I)-np.min(I)) # NB the +0.0 converts to floating point
        start = np.argmax(normI > thresholds[0])
        stop = np.argmax(normI > thresholds[1])

        # Fit a line and find the point where it crosses 0.5
        
        if start==stop:
            transition_x=start
        
        else: 
            gradient, intercept = np.polyfit(x[start:stop], normI[start:stop], 1)
            transition_x = (0.5 - intercept)/gradient # 0.5 = intercept + gradient*xt
        
        # print(f'transition_x:{transition_x}')
        
        # Now, crop out a region centered on the transition
        # start = np.argmax(x > transition_x - width/2.0)
        # stop = np.argmax(x > transition_x + width/2.0)
        start=np.argmin(abs(x-(transition_x - width/2.0)))
        stop=np.argmin(abs(x-(transition_x + width/2.0)))
        # print(f'start:{start} stop:{stop}')
        
        # stop = start + width//row_bayer.shape[0] # Should do the same as above, but guarantees length.
        aligned_rows.append((x[start:stop] - 1*transition_x, I[start:stop], transition_x))
    return aligned_rows


def sorted_x_and_I(aligned_rows):
    """Extract all x and I coordinates from a set of rows, in ascending x order"""
    # First, extract the x and I coordinates into separate arrays and flatten them.
    # xs, Is, txs = zip(*aligned_rows)

    xs=[]
    Is=[]
    tx=[]
    fixed_length=aligned_rows[0][0].shape[0]
    for ar in aligned_rows:
        x,I,t=ar
        if len(x)==fixed_length:
            xs.append(x)
            Is.append(I)
            tx.append(t)
    all_x = np.array(xs).flatten()
    all_I = np.array(Is).flatten()
        

    # Points must be sorted in ascending order in x
    order = np.argsort(all_x)
    sorted_x = all_x[order]
    sorted_I = all_I[order]

    # If any points are the same, spline fitting fails - so add a little noise
    while np.any(np.diff(sorted_x) <= 0):
        i = np.argmin(np.diff(sorted_x))
        sorted_x[i+1] += 0.0001 # 0.0001 is in units of pixels, i.e. insignificant.
    return sorted_x, sorted_I

def average_edge_spline(sorted_x, sorted_I, dx=0.05, knot_spacing=2.0, crop=10):
    """Average the edge, using a spline to smooth it.
    
    Returns an array of interpolated x coordinates and an array of correspoinding I values.
    """
    xmin, xmax = np.min(sorted_x), np.max(sorted_x)
    ks = knot_spacing
    spline = LSQUnivariateSpline(sorted_x, sorted_I, np.arange(xmin + ks, xmax - ks, ks))
    sx = np.arange(xmin + crop, xmax - crop, dx)
    return sx, spline(sx)

def average_edge_binning(sorted_x, sorted_I, dx=0.8, crop=10):
    """Bin points by their x values, and average each bin.
    
    sorted_x, sorted_I: data, in order of ascending x (flattened)
    dx: bin width, default=0.4.  Too small, and you get empty bins.
    crop: use this to ignore some pixels at the start/end.
    
    Return: x, I_mean, I_sd
    """
    xmin, xmax = np.min(sorted_x), np.max(sorted_x)
    binned_x = np.arange(xmin + crop, xmax - crop, dx)
    binned_I_mean = np.empty_like(binned_x)
    binned_I_sd = np.empty_like(binned_x)
    w = 2.0
    for i, x in enumerate(binned_x):
        rr = slice(np.argmax(sorted_x > x - w/2.0), 
                   np.argmax(sorted_x > x + w/2.0),)
        binned_I_mean[i] = np.mean(sorted_I[rr])
        binned_I_sd[i] = np.std(sorted_I[rr], ddof=1)
    return binned_x, binned_I_mean, binned_I_sd

def numerical_diff(x, y, crop=0, sqrt=False):
    """Numerically differentiate a vector of equally spaced values, with x
    
    Returns the mid-points of the x coordinates, and the numerically
    differentiated y values (scaled by the difference between x points).
    If a "crop" argument is supplied, it crops either end of the data by
    that amount (in units of x, rather than simply datapoints).
    """
    crop = np.argmax(x > np.min(x) + crop)
    if crop > 0:
        cropped_x = x[crop:-crop]
    mid_x = (cropped_x[1:] + cropped_x[:-1])/2.0

    if sqrt:
        diff_y = (np.diff((y - np.min(y))**0.5)/np.mean(np.diff(x))**0.5)**2
    else:
        diff_y = np.diff(y - np.min(y))/np.mean(np.diff(x))
    if crop > 0:
        diff_y = diff_y[crop:-crop]
        
    return mid_x, diff_y
#%%
# kp = 1.3 * 2 * np.pi/0.53 # for green light
# x = np.linspace(-1, 1, 400)
# dx = np.mean(np.diff(x))
# lsf = np.sinc(kp * x / np.pi) # numpy defines sinc as sin(pi*x)/(pi*x) for some reason...
# edge = np.cumsum(lsf)**2
# recovered_lsf = np.diff([0] + list(edge))
# recovered_lsf /= np.max(recovered_lsf)

# # We can, of course, recover it more or less perfectly by taking a square root
# # (the more or less part is because we get sign errors)
# recovered_lsf_2 = np.diff([0] + list(edge**0.5))**2

# f, ax = plt.subplots(1,2)
# ax[0].plot(x, lsf**2,label='lsf^2')
# ax[1].plot(x, edge)
# ax[0].plot(x, recovered_lsf,label='re_lsf')
# ax[0].plot(x, recovered_lsf_2,label='re_lsf_2')
# ax[0].legend()


# Note that:
# * The recovered line spread function from differentiating the edge is not the same as the true one, because we measure intensity (amplitude squared).  That also means:
#     * The mid-point of the intensity is not the true position of the edge (again because of the squaring)
#     * The "ringing" is only on the bright side of the edge, not the dark side.
# * ``numpy.sinc`` has an extra factor of $\pi$ in it!
# * The FWHM of the recovered lsf is something like $180\,$nm.  That is a little less than half what I'm seeing :(
# * The maths above is correct for *fully coherent* light.  This is correct in the limit of low-NA illumination in the microscope.  Fully incoherent light requires a larger illumination NA than collection NA (i.e. a very impressive oil-immersion condenser).  Realistically, though, this microscope has partially coherent illumination, as it does use a condenser lens with a non-zero NA.
# * See J. W. Goocman, "Introduction to Fourier Optics", third edition, Chapter 6.2, pp135-138
#%% reading and cropping the image
import cv2
# filename=r"C:\Users\imrul\Repo\usaf_analysis-master\image\marsden\20200731-190832-oneshot-image-19500.png"
# filename=r"G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\mtf\20210719_FFImage_White_30mm_USAF_10msISO100_.bmp"
#filename=r"G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\20210922_IMX258Calibration\vlcsnap-2021-09-23-10h00m07s653.png"
# filename=r"G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\2021-01-21_SensorCharacterization\usaf target\AR1335\20220121123600192_0.bmp"
# filename=r"G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\2021-01-21_SensorCharacterization\usaf target\IMX258\IMX258BF_exp50_analogGain0_vcm350_0..bmp"
# filename=r"G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\2021-01-21_SensorCharacterization\usaf target\IMX477\TIFFs_8bit_auto_brightness\cyclops-R_BF-FOV_0.tiff"
filename=r"Z:\raspberrypi\photos\temp\FAT_Captures\Demo_run000_sample_FAT_disc_v00_Cyc4juno\cyclops-R_BF-FOV_7.png"
pixel_pitch=1.55e-3 # in mm
im=cv2.imread(filename)
im=cv2.medianBlur(im,5)
channel= 2
less_figure=True
edge_smoothing=True

# 
getROI=userROI(im[:,:,channel],no_of_corners=1,sort=False,rectangle=False) # get the roi around the edge
boxSize=[100,60]
roi={'x0':getROI.coords[0][0]-boxSize[0]//2,'x1':getROI.coords[0][0]+boxSize[0]//2,'y0':getROI.coords[0][1]-boxSize[1]//2,'y1':getROI.coords[0][1]+boxSize[1]//2}
# roi={'x0':getROI.coords[0][0],'x1':getROI.coords[1][0],'y0':getROI.coords[0][1],'y1':getROI.coords[1][1]}
# roi={'x0':1500,'x1':1600,'y0':1450,'y1':1550}
# roi={'x0':980,'x1':1130,'y0':930,'y1':1030}
raw_image=np.copy(im[roi['y0']:roi['y1'],roi['x0']:roi['x1']]) # provide the cropped image of the slanted edge
# vertical, falling, line = locate_edge(raw_image)
# raw_image = reorient_image(raw_image, vertical, falling)


raw_image_b2w=check_b2w_transition(raw_image,channel)


plt.figure()
plt.imshow(raw_image_b2w[:,:,channel])
plt.title('Cropped Edge after b2w check')

#%%



#%% extracting edges and lining them up
aligned_rows = extract_aligned_edge(raw_image_b2w, width=raw_image.shape[1], channel=channel)
txs = np.array([tx for x, I, tx in aligned_rows])

f, ax = plt.subplots(1,2, sharex=True, sharey=True)
for x, I, tx in aligned_rows:
    # Manually add back in the transition point shift to get un-aligned rows (!)
    ax[0].plot(x + tx - np.mean(txs), I)
    ax[1].plot(x, I)
#%%
f, ax = plt.subplots(1,2, sharex=True, sharey=True)

# aligned_rows = extract_aligned_edge(raw_image, width=100, channel=2)
for x, I, tx in aligned_rows:
    # Plot the aligned rows in the background
    ax[0].plot(x, I, color="gray")
    
sorted_x, sorted_I = sorted_x_and_I(aligned_rows)
spline_x, spline_I = average_edge_spline(sorted_x, sorted_I)
binned_x, binned_I_mean, binned_I_sd = average_edge_binning(sorted_x, sorted_I,dx=1)
ax[0].plot(spline_x, spline_I, linewidth=2, color="black")
ax[0].set_title('Spline')

ax[1].fill_between(binned_x, binned_I_mean - binned_I_sd, binned_I_mean + binned_I_sd, color="gray")
ax[1].plot(binned_x, binned_I_mean, linewidth=2, color="black")
ax[1].set_title('Binning')
#%%
f, ax = plt.subplots(1,2, sharex=True, sharey=False)
ax[0].plot(*numerical_diff(binned_x, binned_I_mean, crop=5, sqrt=False))
ax[0].plot(*numerical_diff(spline_x, spline_I, crop=5, sqrt=False))
ax[1].plot(*numerical_diff(binned_x, binned_I_mean, crop=5, sqrt=True))
ax[1].plot(*numerical_diff(spline_x, spline_I, crop=5, sqrt=True))
ax[0].set_title('SQRT:FALSE')
ax[1].set_title('SQRT:TRUE')

#%% get the PSF
if edge_smoothing==True:
    smooth_pt=0.5/100 # smoothing sigma as a percentage of total length
    plt.figure(401)
    plt.plot(spline_x,spline_I,'k--',label='Original')
    spline_I=gaussian_filter1d(spline_I,len(spline_I)*smooth_pt)
    plt.plot(spline_x,spline_I,label=f'Smoothed {smooth_pt*100} %')
    plt.xlabel('Horizontal position [px]')
    plt.ylabel('Pixel Intensity [a.u.]')
    plt.legend()

# plt.figure()
# plt.plot(spline_x,spline_I)
# plt.plot(spline_x,sI)


spline_I_normalized=(spline_I-np.min(spline_I))/(np.max(spline_I)-np.min(spline_I))




xx,yy=numerical_diff(spline_x, spline_I, crop=10, sqrt=False)
# xx,yy=numerical_diff(binned_x, binned_I_mean, crop=10, sqrt=False)
# yy=abs(yy)  # Forcing negative PSF to be positive this may be an invalid assumption. only useful when sqrt=False
# yy=yy**2  # point of concern
plt.figure()
plt.plot(xx,yy/np.max(yy))

#%% getting image height of the ROI
edge_coords=np.array([np.mean([roi['x0'],roi['x1']]),np.mean([roi['y0'],roi['y1']])])
h,w=im.shape[:-1]
mid_coords=np.array([w/2,h/2])

temp=edge_coords-mid_coords

imRadius=np.sqrt(temp[0]**2+temp[1]**2)
maxRadius=np.sqrt(mid_coords[0]**2+mid_coords[1]**2)

correction_theta=0
if temp[0]<0:
    correction_theta=180

imHeight_pt=np.round(imRadius/maxRadius*100,1)
imTheta=np.round(np.degrees(math.atan(-temp[1]/temp[0])),2)+correction_theta
imTheta=np.round(imTheta)



#%% centering the peak
yy=yy/np.max(yy)
peaks, _ = scipy.signal.find_peaks(yy, height=0.9)
plt.figure()
plt.plot(yy)
plt.plot(peaks, yy[peaks], "x")
plt.show()

shift=np.argmin(abs(xx-0))-peaks

yy_shifted=np.roll(yy,shift)


plt.figure(400)
# plt.plot(spline_x,spline_I,'k--',label='Original')
# spline_I=gaussian_filter1d(spline_I,len(spline_I)*smooth_pt)
plt.plot(spline_x,spline_I_normalized,label=f'imH(%),imT(deg):{imHeight_pt},{imTheta}')
plt.xlabel('Horizontal position [px]')
plt.ylabel('Pixel Intensity [a.u.]')
plt.legend()
plt.title('Normalized Cross section of subsampled sEdge')
plt.grid(True)


plt.figure(500)
plt.plot(xx,yy_shifted,label=f'imH(%),imT(deg):{imHeight_pt},{imTheta}')
# plt.plot(xx,yy,label='original')
plt.grid(True)
plt.legend()
plt.xlim([-25,25])
signal_lsf=np.copy(yy_shifted)
plt.title('Line Spread Function')
plt.xlabel('Pixel position (px)')
plt.ylabel('Intensity [a.u.]')

#%% get the MTF





nSample=len(xx)
dx=np.mean(np.diff(xx))
maxFreq=0.5/dx
dFreq=2*maxFreq/(nSample-1)
ff=np.linspace(-nSample*dFreq,nSample*dFreq,nSample) # cy/px

ff_lpmm=ff/pixel_pitch  # lp/mm



mtf=np.abs(fftshift(fft(signal_lsf)))

normMTF=mtf/np.max(mtf)



#%% shift the MTF curve so that peak aligns to DC component
# peaks, _ = scipy.signal.find_peaks(normMTF, height=0.99)
# shift=np.argmin(abs(ff_lpmm-0))-peaks
# if len(shift)>1:
#     shift=shift[0]

# if abs(shift)>=2:
#     normMTF_shifted=np.roll(normMTF,shift)
# else:
#     normMTF_shifted=np.copy(normMTF)

# # plt.figure()
# # plt.plot(ff_lpmm,normMTF,'o-',label='original')
# # plt.plot(ff_lpmm,normMTF_shifted,'o-',label='shifted')
# # plt.legend()
# # plt.xlim([-200,200])
# # plt.grid()

# # normMTF_original=np.copy(normMTF)
# # normMTF=np.copy(normMTF_shifted)

#%% print MTF50 value with image height

ff_lpmm_ip=np.linspace(ff_lpmm[0],ff_lpmm[-1],10*len(ff_lpmm)) # interpolation for granularity

normMTF_ip=np.interp(ff_lpmm_ip,ff_lpmm,normMTF) 

l=len(normMTF_ip)
index1=np.argmin(abs(normMTF_ip[:l//2]-0.5))
index2=np.argmin(abs(normMTF_ip[l//2:]-0.5))
print(ff_lpmm_ip[index1])
print(ff_lpmm_ip[l//2+index2])

mtf50=np.round(np.mean([abs(ff_lpmm_ip[index1]),(ff_lpmm_ip[l//2+index2])]),2)



print(f'Image Height: {imHeight_pt} % \nImage Theta: {imTheta} deg')
print(f'MTF50: {mtf50} lp/mm')


#%%

peaks, _ = scipy.signal.find_peaks(normMTF_ip, height=0.95)
shift=np.argmin(abs(ff_lpmm_ip-0))-peaks

if len(shift)>1:
    shift=shift[0]

if np.abs(shift)>=2:
    normMTF_shifted=np.roll(normMTF_ip,shift)
else:
    normMTF_shifted=np.copy(normMTF_ip)
plt.figure()
plt.plot(ff_lpmm_ip,normMTF_ip,'o-',label='original')
plt.plot(ff_lpmm_ip,normMTF_shifted,'o-',label='shifted')
plt.legend()
plt.xlim([-200,200])
plt.grid()    

#%% plot the shifted and normalized MTF
plt.figure(600)
plt.plot(ff_lpmm_ip,normMTF_shifted,label=f'imH(%),imT(deg):{imHeight_pt},{imTheta}')
plt.plot(ff_lpmm_ip,0.5*np.ones(np.shape(normMTF_ip)),'k--')
plt.grid(True)
plt.xlim([-1,200])
plt.xlabel('Spatial Frequency lp/mm')
plt.ylabel('Normalized Contrast [%]')
plt.title('Modulation Transfer Function (MTF)')
plt.legend()
#%% fig processing
if less_figure is True:
    
    figNum=[400,500, 600]
    for i in plt.get_fignums():
        if i not in figNum:
            plt.close(i)
            
#%%
# plt.figure()
# plt.plot(Valuescsv[:,0],Valuescsv[:,1])
# plt.plot(Valuescsv[:,0],0.5*np.ones(np.shape((Valuescsv[:,0]))))
# plt.grid(True)
# plt.xlabel('Spatial Frequency cy/mm')
# plt.ylabel('Normalized Contrast [%]')
# plt.title('MTF from imageJ')
            