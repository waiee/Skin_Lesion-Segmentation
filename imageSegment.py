# -*- coding: utf-8 -*-
"""
imageSegment.py

YOUR WORKING FUNCTION

"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

#Uncomment to use main dataset
input_dir = 'dataset/test'
output_dir = 'dataset/output'
groundtruth_dir = 'dataset/groundtruth'

#Uncomment to use additional dataset
# input_dir = 'addDataset/AddTest'
# output_dir = 'addDataset/output'
# groundtruth_dir = 'addDataset/AddTestGT'

# you are allowed to import other Python packages above
#########################
def segmentImage(img):
    # Inputs
    # img: Input image, a 3D numpy array of row*col*3 in BGR format
    #
    # Output
    # outImg: segmentation image
    #
    #########################################################################
    # ADD YOUR CODE BELOW THIS LINE

    # grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply Gaussian blur
    blur_img = cv2.GaussianBlur(gray_img, (31, 31), 0)

    # global thresholding
    _, th = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # morphological operations
    kernel = np.ones((10, 10), np.uint8)
    closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=7)
    
    # apply Canny edge detector
    edges = cv2.Canny(closing, 100, 200)

    # find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours by area and remove small ones
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # approximate contours to smooth them
    approx_contours = [cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True) for cnt in contours]

    # create an empty mask to draw the refined contours
    outImg = np.zeros_like(gray_img)
    
    # draw the largest contour (assuming it's the lesion)
    cv2.drawContours(outImg, [approx_contours[0]], 0, (255), thickness=cv2.FILLED)

    # normalize the image intensity
    outImg = cv2.normalize(outImg, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # END OF YOUR CODE
    #########################################################################
    return outImg

###################### UNCOMMENT TO SEE RESULT IMMEDIATELY ##############################
# print("All images has been segmented successfully. Loading result...")
# # Uncomment to use main dataset: load original image & ground truth image
# originalImg = cv2.imread(input_dir + '/'+ 'SL_001.jpg')
# gtImg = cv2.imread(groundtruth_dir+ '/'+'SL_GT_001.png', cv2.IMREAD_GRAYSCALE)

# # Uncomment to use additional dataset: load original image & ground truth image
# # originalImg = cv2.imread(input_dir + '/'+ 'SLA_01.jpg')
# # gtImg = cv2.imread(groundtruth_dir+ '/'+'SLA_GT_01.png', cv2.IMREAD_GRAYSCALE)

# # convert BGR to RGB
# originalImg_rgb = cv2.cvtColor(originalImg, cv2.COLOR_BGR2RGB)

# # segmentation function
# segmented_image = segmentImage(originalImg)

# # visualization
# plt.figure(figsize=(10, 5))

# plt.subplot(1, 3, 1)
# plt.imshow(originalImg_rgb)
# plt.title('Original Image')
# plt.xticks([]), plt.yticks([])

# plt.subplot(1, 3, 2)
# plt.imshow(gtImg, cmap='gray')
# plt.title('Ground Truth')
# plt.xticks([]), plt.yticks([])

# plt.subplot(1, 3, 3)
# plt.imshow(segmented_image, cmap='gray')
# plt.title('Segmented Image')
# plt.xticks([]), plt.yticks([])
# plt.show()
