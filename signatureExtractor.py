###############################
#
#   (c) Vlad Zat 2017
#   Student No: C14714071
#   Course: DT228
#   Date: 06-10-2017
#
#	Title: Signature Extractor
#
#   Introduction:
#
#   Describe the algorithm here instead of doing it line by line (10 - 15 lines)
#	Preferably use a systematic approach
#	e.g. step-by-step
#	or pseudocode
#	Give an overview
#	Comment on experiments
#	Use references (Harvard Reference System or IEEE) - no weblinks

# Get Signature from Page:
# convert image to grayscale
# get edges of the signature
# close the result
# find contours
# find the contour with the biggest bounding rectangle area
# add padding to the bounding rectangle
# generate a new image that only contains the largest bounding rectangle

# Extra Notes:
# Filtering is not necessary because writting doesn't have a texture to impede edge detection
# Filtering in this case only makes the text harder to read


import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui


class Rect:
    def __init__(self, x=0, y=0, w=0, h=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = 0

    def set_area(self, area):
        self.area = area

    def get_area(self):
        return self.area

    def set(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = w * h

    def add_padding(self, img_size, padding):
        self.x -= padding
        self.y -= padding
        self.w += 2 * padding
        self.h += 2 * padding
        if self.x < 0:
            self.x = 0
        if self.y < 0:
            self.y = 0
        if self.x + self.w > img_size[0]:
            self.w = img_size[0] - self.x
        if self.y + self.h > img_size[1]:
            self.h = img_size[1] - self.y


# TODO Use easygui to open images
images_path = 'images/'
# signature = cv2.imread(images_path + 'Trump.jpg')
signature = cv2.imread(images_path + 'sig3.jpg')

# signature = cv2.imread(imgsPath + 'cs-11-wo-stamp-and-bar.bmp')
# signature = cv2.medianBlur(signature, 3)
# signature = cv2.GaussianBlur(signature, (3, 3), 0)


# TODO Throw error if it's not a valid image
def get_page_from_image(img):
    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # b_img = cv2.medianBlur(src = g_img, ksize = 11)
    b_img = g_img.copy()

    threshold, _ = cv2.threshold(src=b_img, thresh=0, maxval=255, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    canny_img = cv2.Canny(image=b_img, threshold1=0.5 * threshold, threshold2=threshold)

    _, contours, _ = cv2.findContours(image=canny_img.copy(), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    # There is no page in the image
    if len(contours) == 0:
        print('No Page Found')
        return img

    max_rect = Rect(0, 0, 0, 0)
    for contour in contours:
        # Detect edges
        # Reference - http://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
        epsilon = cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, 0.1 * epsilon, True)
        x, y, w, h = cv2.boundingRect(contour)
        current_area = w * h
        # current_area = cv2.contourArea(contour)

        # check if length of approx is 4
        if len(corners) == 4 and current_area > max_rect.get_area():
            max_rect.set(x, y, w, h)
            print(cv2.isContourConvex(contour))
            # max_rect.setArea(current_area)

    contours_in_page = 0
    for contour in contours:
        x, y, _, _ = cv2.boundingRect(contour)
        if (max_rect.x < x < max_rect.x + max_rect.w) and (max_rect.y < y < max_rect.y + max_rect.h):
            contours_in_page += 1

    max_contours = 5
    if contours_in_page <= max_contours:
        print('No Page Found')
        return img

    return img[max_rect.y: max_rect.y + max_rect.h, max_rect.x: max_rect.x + max_rect.w]


def get_signature_from_page(img):
    img_size = np.shape(img)

    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # The values for edge detection can be approximated using Otsu's Algorithm
    # Reference - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.402.5899&rep=rep1&type=pdf
    threshold, _ = cv2.threshold(src=g_img, thresh=0, maxval=255, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    canny_img = cv2.Canny(image=g_img, threshold1=0.5 * threshold, threshold2=threshold)

    # Close the image to fill blank spots so blocks of text that are close together (like the signature) are easier to detect
    # Signature usually are wider and shorter so the strcturing elements used for closing will have this ratio
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(30, 1))
    canny_img = cv2.morphologyEx(src=canny_img, op=cv2.MORPH_CLOSE, kernel=kernel)

    # findContours is a distructive function so the image pased is only a copy
    _, contours, _ = cv2.findContours(image=canny_img.copy(), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    max_rect = Rect(0, 0, 0, 0)
    max_corners = 0

    for contour in contours:
        epsilon = cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, 0.01 * epsilon, True)
        x, y, w, h = cv2.boundingRect(contour)
        current_area = w * h
        # Maybe add w > h ?
        # if current_area > max_rect.getArea():
        if len(corners) > max_corners:
            max_corners = len(corners)
            max_rect.set(x, y, w, h)

    # Increase the bounding box to get a better view of the signature
    max_rect.add_padding(img_size=img_size, padding=10)

    return img[max_rect.y: max_rect.y + max_rect.h, max_rect.x: max_rect.x + max_rect.w]


def get_signature(img):
    img_size = np.shape(img)

    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # minBlockSize = 3
    # maxBlockSize = 101
    # minC = 3
    # maxC = 101
    #
    # bestContourNo = 1000000
    # bestBlockSize = 0
    # bestC = 0
    #
    # for c in range(minC, maxC, 2):
    #     for bs in range(minBlockSize, maxBlockSize, 2):
    #         mask = cv2.adaptiveThreshold(g_img, maxValue = 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType = cv2.THRESH_BINARY, block_size = bs, c = c)
    #         rmask = cv2.bitwise_not(mask)
    #         _, contours, _ = cv2.findContours(image = rmask.copy(), mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)
    #         if len(contours) > 15 and len(contours) < bestContourNo:
    #             bestContourNo = len(contours)
    #             bestBlockSize = bs
    #             bestC = c

    # block_size = 21, c = 10

    # TODO throw error if block_size is bigger than image
    block_size = 21
    c = 10
    if block_size > img_size[0]:
        if img_size[0] % 2 == 0:
            block_size = img_size[0] - 1
        else:
            block_size = img_size[0]

    if block_size > img_size[1]:
        if img_size[0] % 2 == 0:
            block_size = img_size[1] - 1
        else:
            block_size = img_size[1]

    mask = cv2.adaptiveThreshold(g_img, maxValue=0, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                 thresholdType=cv2.THRESH_BINARY, blockSize=block_size, C=c)
    rmask = cv2.bitwise_not(mask)

    return cv2.bitwise_and(signature, signature, mask=rmask)


# Camera capture
# camera = cv2.VideoCapture(0)
# (grabbed, signature) = camera.read()
# cv2.imshow('Picture', signature)
# cv2.waitKey(0)


signature = get_page_from_image(img=signature)
# cv2.imshow('1', signature)

signature = get_signature_from_page(img=signature)
cv2.imshow('2', signature)

signature = get_signature(img=signature)
cv2.imshow('Signature', signature)

key = cv2.waitKey(0)

# def adaptiveThreshold(img):
#     g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # TODO remove the constrants from here:
#     img = cv2.adaptiveThreshold(g, maxValue = 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType = cv2.THRESH_BINARY, blockSize = 21, C = 10)
#     # img = cv2.adaptiveThreshold(g, maxValue = 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType = cv2.THRESH_BINARY, blockSize = 21, C = 10)
#     return img
#
# mask = adaptiveThreshold(signature)
# # k = np.array([[1,4,1], [4,7,4], [1,4,1]], dtype=float)
# # blurredSignature = cv2.filter2D(mask,-1,k)
#
# # im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # cv2.drawContours(signature, contours, -1, (0,255,0), 3)
#
# # TODO Clean the mask
# # shape = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
# # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, shape)
#
# rmask = cv2.bitwise_not(mask)
# # shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# # rmask = cv2.morphologyEx(rmask, cv2.MORPH_CLOSE, shape)
# signature = cv2.bitwise_and(signature, signature, mask=rmask)
#
# # TODO ??? Click on an image to append the signature there
#
# # Reference - http://docs.opencv.org/3.3.0/d4/d13/tutorial_py_filtering.html
# k = np.array([[1,4,1], [4,7,4], [1,4,1]], dtype=float)
# blurredSignature = cv2.filter2D(signature,-1,k)
# imgs = np.hstack((signature, blurredSignature))
#
# sHeight, sWidth, _ = signature.shape
# scaledSignature = cv2.resize(signature, (2 * sWidth, 2 * sHeight))
