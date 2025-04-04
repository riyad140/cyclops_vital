# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 10:29:22 2020

@author: imrul.kayes
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from matplotlib.patches import Rectangle, Polygon, Circle
import sys
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


class userROI():
    # This class allows user to select an arbritary polygon

    def __init__(self, ims, no_of_corners=4, sort=True, rectangle=False):

        if ims.shape[-1] == 3:
            ims = cv.cvtColor(ims, cv.COLOR_BGR2GRAY)
        self.no_of_corners = no_of_corners
        self.sort = sort
        self.rectangle = rectangle

        if self.no_of_corners != 2 and self.rectangle == True:
            print('WARNING: You must have no_of_corners=2 if you want to have a rectangular ROI\n Stopping the script')
            sys.exit()

        print('#################################################################')
        print('Select the %d corners of the intended ROI' % no_of_corners)
        print('#################################################################')
        self.printInstructions()
        self.fig, self.axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
        self.matshow(ims, vmin=0, vmax=np.max(ims))
        self.coords = []
        self.patches = []
        self.cidk = self.fig.canvas.mpl_connect('key_press_event', self.onKey)
        self.imShape = ims[0].shape
        plt.show(block=False)
        self.fig.canvas.start_event_loop()  # block here until

    def matshow(self, mat, cmap='gray', vmin=0, vmax=1):
        if len(mat.shape) == 2:
            self.axes.matshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        elif len(mat.shape) == 3:
            self.axes.matshow(mat)
        else:
            raise AttributeError(f'Mat has shape {mat.shape}. Must be of shape [rows, cols] or [rows, cols, channels]')

    def printInstructions(self):
        print("INSTRUCTIONS:")
        print("Build an ROI by:")
        print("Press spacebar, then clicking the corners in a clockwise direction")
        print("Or, if you are selecting a single point, Just press spacebar, click on the image")
        print("Press enter if you are satisfied with the drawn ROI")

        print("Alternatively, just press \"a\" if you want to select all the image")

    def onclick(self, event):
        ix, iy = event.xdata, event.ydata

        self.coords.append([int(ix), int(iy)])

        if len(self.coords) > 0 and len(self.coords) < self.no_of_corners:
            self.fig.canvas.mpl_disconnect(self.cid)

        if len(self.coords) == self.no_of_corners:
            self.fig.canvas.mpl_disconnect(self.cid)
            if self.sort == True:
                self.sortCorners()

            self.showROI()

    def onKey(self, event):
        count = 0
        if event.key == " ":
            if len(self.coords) < self.no_of_corners and len(self.coords) > 0:
                count = count + 1
                print("Select the next corner in a clockwise direction ")

            else:
                self.coords = []
                print("Getting 1st corner coordinates")
            self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
    #
        if event.key == "enter":
            if len(self.coords) == self.no_of_corners:
                print("You Selected Valid ROI")
                if self.sort != True:
                    print("Not Sorted")
                print(self.coords)
                self.fig.canvas.stop_event_loop()
                self.fig.canvas.mpl_disconnect(self.cidk)

            else:
                print("Error: You need to choose %d points" % self.no_of_corners)

        if event.key == "a":
            print("Using all frame")
            self.coords = [[0, 0], [self.imShape[1], self.imShape[0]]]
            self.fig.canvas.stop_event_loop()
            self.fig.canvas.mpl_disconnect(self.cidk)
            plt.close(self.fig)

    def showROI(self):
        for patch in self.patches:
            patch.remove()
            self.patches = []

        if self.rectangle == True:
            self.patches.append(self.axes.add_patch(Rectangle((self.coords[0][0], self.coords[0][1]), self.coords[1][0] - self.coords[0][0], self.coords[1][1] - self.coords[0][1], edgeColor="r", fill=False)))
            print('Drawing a Rectangle instead of line or polygon. To turn it off, set rectangle=False')
        elif self.no_of_corners == 1:
            self.patches.append(self.axes.add_patch(Circle((self.coords[0][0], self.coords[0][1]), 2, edgeColor="r", fill=True)))

        else:
            self.patches.append(self.axes.add_patch(Polygon(self.coords, edgeColor="r", fill=False)))
        self.fig.canvas.draw()

    def sortCorners(self):
        corners = self.coords
        center = np.mean(corners, axis=0)
        top = []
        bot = []
        for corner in corners:
            if (corner[1] < center[1]):
                top.append(corner)
            else:
                bot.append(corner)

        tl = top[1] if top[0][0] > top[1][0] else top[0]
        tr = top[0] if top[0][0] > top[1][0] else top[1]
        bl = bot[1] if bot[0][0] > bot[1][0] else bot[0]
        br = bot[0] if bot[0][0] > bot[1][0] else bot[1]

        self.coords = np.array((tl, tr, br, bl), np.float32)


# %%
if __name__ == '__main__':
    filename = r"C:\Users\imrul.kayes\AiryAppCaptures\2021-05-19_depthMetric_benchmarking\13-38-57\HM41_spillover_[30,50]cm_Demosaic_image_00000.tiff"
    imageSum = cv.imread(filename, 0)
    HH = userROI(imageSum, no_of_corners=2, sort=False, rectangle=True)
