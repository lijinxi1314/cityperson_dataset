from scipy import ndimage
import numpy as np
import cv2
from skimage import data, feature, color, filters, img_as_float
from skimage.filters import roberts, sobel
from matplotlib import pyplot as plt

def L(input):
    input = cv2.cvtColor(input, cv2.COLOR_BGR2Luv)
    l, u, v = cv2.split(input)
    return l


def U(input):
    input = cv2.cvtColor(input, cv2.COLOR_BGR2Luv)
    l, u, v = cv2.split(input)
    return u

def V(input):
    input = cv2.cvtColor(input, cv2.COLOR_BGR2Luv)
    l, u, v = cv2.split(input)
    return v


def Gray(input):
    input=cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    return input

def Threshold(input):
    # cv2.threshold(input,127,255,cv2.THRESH_BINARY)
    input=cv2.adaptiveThreshold(input, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    return input



def DoG(input,num):
    k = 1.6
    if(num==1):sigma=4.0
    if(num==2):sigma=8.0
    if(num==3):sigma=16.0

    input = img_as_float(input)
    input = color.rgb2gray(input)
    s1 = filters.gaussian(input,k*sigma)
    s2 = filters.gaussian(input,sigma)
    dog = s1 - s2
    return dog


def Gm(input,num):
    if num==1:
        output = cv2.Sobel(input,cv2.CV_64F,1,0,ksize=3)
    if num==2:
         output = cv2.Sobel(input,cv2.CV_64F,0,1,ksize=3)
    return output


def Canny(input):
    # input = img_as_float(input)
    # input = color.rgb2gray(input)
    # output= feature.canny(input, sigma=1)
    input=cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    output=cv2.Canny(input,150,200)
    return output


#################################################################################
# cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, phi, ktype)
# ksize - size of gabor filter (n, n) --> line width of road markings in pixel
# sigma - standard deviation of the gaussian function
# theta - orientation of the normal to the parallel stripes
# lambda - wavelength of the sunusoidal factor
# gamma - spatial aspect ratio
# phi - phase offset
# ktype - type and range of values that each pixel in the gabor kernel can hold
################################################################################
def Garbor(input,num):
    input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    theta=[0,45,90,135,180,225,270,315]
    g_kernel = cv2.getGaborKernel((9, 9), 4.5, theta[num], 8.0,10, 0, ktype=cv2.CV_32F)
    # if num==1:
    #     g_kernel = cv2.getGaborKernel((9, 9), 4.5, theta[num], 8.0,10, 0, ktype=cv2.CV_32F)
    filtered_img = cv2.filter2D(input, cv2.CV_8UC3, g_kernel)

    return filtered_img


def Robinson(input, direction):
    input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    input=np.asarray( input, dtype=np.float64 )
    if direction=="robinson_east":
        roberts_cross_v = np.array( [[ -1.0, -2.0, -1.0 ],
                             [ 0.0,0.0,0.0 ],
                             [ 1.0,2.0,1.0 ]])

    if direction=="robinson_north":
        roberts_cross_v = np.array( [[ -1.0, 0.0, 1.0 ],
                             [ -2.0,0.0,2.0 ],
                             [ -1.0,0.0,1.0 ]])

    if direction=="robinson_northeast":
        roberts_cross_v = np.array( [[  -2,-1,0 ],
                             [ -1,0,1 ],
                             [  0,1,2  ]])

    if direction=="robinson_northwest":
        roberts_cross_v = np.array( [[0,1,2 ],
                             [ -1,0,1 ],
                             [-2,-1,0 ]])

    if direction=="robinson_south":
        roberts_cross_v = np.array( [[1,0,-1 ],
                             [  2,0,-2],
                             [1,0,-1 ]])

    if direction=="robinson_southeast":
        roberts_cross_v = np.array( [[ 0,-1,-2 ],
                             [ 1,0,-1 ],
                             [ 2,1,0 ]])

    if direction=="robinson_southwest":
        roberts_cross_v = np.array( [[ 2,1,0 ],
                             [ 1,0,-1  ],
                             [ 0,-1,-2 ]])

    if direction == "robinson_west":
        roberts_cross_v = np.array( [[ 1,2,1 ],
                             [  0,0,0 ],
                             [  -1,-2,-1 ]])

    vertical = ndimage.convolve( input, roberts_cross_v )
    return vertical
