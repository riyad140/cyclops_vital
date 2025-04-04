# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 21:29:28 2021

@author: imrul
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
#%%
class SimpleISP:
    def __init__(self, imPath='',imageSize = (1871, 2880), header_row=5 ,trailer_row=6,imName='' ):
        npimg = np.fromfile(imPath, dtype=np.uint16)   # read raw image from file      
        imRaw_ = npimg.reshape(imageSize)       
        imRaw=imRaw_[header_row:-trailer_row,:]        # removing header and trailer rows
        # plt.figure()
        # plt.imshow(imRaw)
        self.imRaw=imRaw
        self.imageSize=imageSize        
        self.imName=imName

    def linearize_companded_image_data(self,buffer: np.ndarray, pedestal: int = 240) -> np.ndarray:
        """
        24-bit companded input to linear float32
        """
        # Knee points raw -> companded:
        _knee_points_in = [0, 3904, 287152, 2238336, 16777200]
        # Matching companded knee points:
        _knee_points_out = [0, 3904, 23520, 54416, 65280]
        buffer = buffer.astype(np.float32)
        # In principle, we may get negative values in linear light
        # (keeping them is good for noise cancellation in the shadows):
        buffer -= pedestal
        decompanded = np.zeros_like(buffer).astype(np.float32)
        # Below lowest knee point (0), use same as just above:
        mask = (buffer <= _knee_points_out[0])
        decompanded[mask] = (_knee_points_in[0] +
        ((_knee_points_in[1] - _knee_points_in[0]) /
        (_knee_points_out[1] - _knee_points_out[0]))
        * (buffer[mask] - _knee_points_out[0]))
        for i in range(len(_knee_points_out) - 1):
            mask = ((_knee_points_out[i] <= buffer) &
            (buffer < _knee_points_out[i + 1]))
            decompanded[mask] = (_knee_points_in[i] +
            ((_knee_points_in[i + 1] - _knee_points_in[i]) /
            (_knee_points_out[i + 1] - _knee_points_out[i]))
            * (buffer[mask] - _knee_points_out[i]))
        decompanded[buffer >= _knee_points_out[-1]] = _knee_points_in[-1]
        return decompanded
   
    def get_linear_Raw_image(self):
        # print(f'{self.real}+{self.imag}j')
        imRaw_linear=self.linearize_companded_image_data(self.imRaw)
        return imRaw_linear
    
    def demosaic(self,bit_depth=16): # Edge aware debayering to remove the effect of CFA
        # Assuming Flatfield correction is not necessary, first step is to seperate 3 color channels and use edge aware ...
        # interpolation to find missing values. And then normalizing the image to boost local contrast
        maxVal=2**bit_depth-1
        imBGR=cv2.demosaicing(	self.imRaw,  cv2.COLOR_BayerRG2BGR_EA)
        img_scaled = cv2.normalize(imBGR, dst=None, alpha=0, beta=maxVal, norm_type=cv2.NORM_MINMAX) # Normalizing to enhance contrast
        imRGB=(img_scaled/maxVal).astype(np.float32)
        return imRGB
    
    def autoExposure(self,targetIntensity=0.5):
        # Second step is to adjust the average intensity level of the grayscale image so that it stays at the middle of the ...
        # sensor dynamic range
        imGray=cv2.cvtColor(self.imRGB, cv2.COLOR_BGR2GRAY)
        hist=np.histogram(imGray.flatten())
        ind=np.argmax(hist[0])
        meanIntensity=hist[1][1+ind]
        digital_gain=(targetIntensity/meanIntensity)
        self.digital_gain=digital_gain
        img_exposed=(self.imRGB.astype(float)*digital_gain).clip(0,1).astype(np.float32)
        return img_exposed
    
    def deNoiser(self,sigmaD=10,sigmaI=0.05):
        # Third step is to apply a denosing algorithm to remove high frequency sensor noise. Bilateral filter does this without blurring edges
        bilateral = cv2.bilateralFilter(self.imAE, 10, 0.05, 10).clip(0,1).astype(np.float32)
        return bilateral
    
    def autoWhiteBalance(self, percentile_value=99): 
        # Normalize each color channel by its percentile_value to remove the effect of illumination source
        image=self.imDN
        whitebalanced =( ((image*1.0 / np.percentile(image,percentile_value, axis=(0, 1))).clip(0, 1)).astype(np.float32)) 
        return whitebalanced
    
    def gammaAdjustment(self,gamma=1.0):
        # To apply nonlinearity in the image to enhance brighter OR darker region of interest
        bit_depth=8
        image=(self.imAWB*(2**bit_depth-1)).astype(np.uint8)
        invGamma = 1.0 / gamma
        table = np.array([((i / ((2**bit_depth-1))) ** invGamma) * (2**bit_depth-1)
  		for i in np.arange(0, 2**bit_depth)]).astype("uint8")        
        return cv2.LUT(image, table)
        
    def runISP(self):   
        # Putting it altogether
        self.imLINEAR=self.linearize_companded_image_data(self.imRaw)# Getting Linear RAW image
        self.imRGB=self.demosaic()# debayering companded RAW image into RGB image
        self.imAE=self.autoExposure()# Adjusting brightness or effective exposure
        self.imDN=self.deNoiser() # Removing sensor noise 
        percentile_value=99
        gamma=1.2
        
        if self.digital_gain>5:## highlighting either brighter or darker area depending on effective light level of the scene
            percentile_value=99.9 
        elif self.digital_gain<1.5:
            gamma=0.8
            
        self.imAWB=self.autoWhiteBalance(percentile_value=percentile_value) # White balancing
        self.im=self.gammaAdjustment(gamma=gamma) # Gamma corrected 8 bit output for display        
        plt.figure()# display the image
        plt.imshow(self.im) 
        plt.title(self.imName)
 #%%   
    
if __name__ == "__main__":
    
    binPath=r"C:\Users\imrul\Downloads\ISP_Programming_Assessment\ISP_Programming_Assessment\images" # path to raw files
    for file in os.listdir(binPath):
        if file.endswith('.raw'):
            print(f'Processing---------------------- {file}')
            filePath=os.path.join(binPath,file)
            ispEngine=SimpleISP(filePath,imName=file)
            ispEngine.runISP()
    
                    
        
        
        
        
        
        
        
    