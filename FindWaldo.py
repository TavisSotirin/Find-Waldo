from skimage import img_as_ubyte, io
from skimage.feature import canny

import cv2
import numpy as np
import math

def CircleDistance(circles, minDist, maxDist, angleThresh):
    if circles is not None:
        distance = np.zeros((len(circles),len(circles)), np.float64)
        valid = [[False for x in range(len(circles))] for x in range(len(circles))]
        ignore = []
        safe = []
        
        for i,c1 in enumerate(circles):
            for j,c2 in enumerate(circles):
                if i != j:
                    xDist = abs(c1[0] - c2[0])
                    yDist = abs(c1[1] - c2[1])
                    
                    dist = math.sqrt(xDist ** 2 + yDist ** 2)
                    
                    if yDist != 0:
                        valid[i][j] = (xDist / yDist > angleThresh) and dist > minDist and dist < maxDist
                    
                    if dist < minDist and j not in ignore and j not in safe:
                        ignore.append(j)
                        safe.append(i)
                    
                    distance[i][j] = dist
        
        for i in ignore:
            for j in range(len(valid[i])):
                valid[i][j] = False
        
        return distance, valid
    
def Hough(image, mask, imname, radHi = 22, radLo = 12, minDist = 10, maxDist = 30, angleThresh = 1):
    # Keep a copy of original image
    im = image.copy()
    
    # Output binary mask and reopen to appease HoughCircles
    io.imsave("tmp.jpg", img_as_ubyte(mask))
    gray = cv2.cvtColor(cv2.imread('tmp.jpg', cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
    #50, 18 -- 60,18 works better
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=10,
                     param1=65, param2=19, minRadius=radLo, maxRadius=radHi)

    if circles is not None:
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(im, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(im, (int(i[0]), int(i[1])), 2, (0, 0, 255), 3)
    
    im2 = image.copy()
    
    # Find distances between circles and remove invalid ones from consideration
    if circles is not None:
        distance, valid = CircleDistance(circles[0], minDist, maxDist, angleThresh)
        
        validCircles = []
        for i,row in enumerate(valid):
            for j,valid in enumerate(row):
                if valid and i not in validCircles:
                    validCircles.append(i)
        
        outCirc = []
        outDist = []
        for i in validCircles:
            outCirc.append(circles[0][i])
            outDist.append(distance[i])
        
        for c in outCirc:
            cv2.circle(im2, (int(c[0]), int(c[1])), int(c[2]), (0, 0, 255), 2)
        
        if len(np.array(outCirc)) > 1:
            io.imsave("output/" + imname + "_ALL.jpg", im)
            io.imsave("output/" + imname + "_SC.jpg", im2)
        
        return np.array(outCirc), np.array(outDist)
    
    else:
        return [],[]
    
def DetectCircles(image, imname):
    # Scale up to enhance smaller circles pixel count
    image = cv2.resize(image, (image.shape[1] * 5, image.shape[0] * 5))
    
    im_dark = image.copy()
    # Grey band image to 3 weighted bands to clean noise and accentuate blacks, works well since images are pretty light
    b1 = image < 110
    b2 = image < 210
    b3 = image < 256
    
    b3 = np.logical_xor(b3,b2)
    b2 = np.logical_xor(b2,b1)
    
    im_dark[b1] = 0
    im_dark[b2] = 127
    im_dark[b3] = 255
    
    # Sharpen image twice
    sharp_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) / 9
    img_sharp = cv2.filter2D(im_dark, -1, sharp_kernel)
    img_sharp = cv2.filter2D(img_sharp, -1, sharp_kernel)
    
    # Apply to initial image
    img_sharp = img_sharp - image
    
    # Get Canny edges of sharpened image
    sharp_edges = canny(img_sharp, sigma=1, low_threshold=1, high_threshold=200)
    
    #io.imsave("output/edges/" + imname + "_edge.jpg", sharp_edges)
    
    # Run Hough transform to find valid circles on edges
    circ, circDist = Hough(image, sharp_edges, imname, radHi = 22, radLo = 12, minDist = int(image.shape[1] * .09), maxDist = int(image.shape[1] * .15), angleThresh = 2.9)
    
    # If we found 2 circles we have a potential match
    return len(circ) > 1 

def ContourBounds(image, mask, imShape, kernelSize, scale_x = 1.5, scale_y = 2.5, iterations = 6):
    # Calculate smaller kernel size at 40% original kernel size
    small = int(kernelSize * 0.4) if int(kernelSize * 0.4) % 2 == 1 else int(kernelSize * 0.4) - 1
    small = small if small > 2 else 3
    
    # Create horizontal kernel to dilate mask horizontally
    small = small + 0
    kernel = np.zeros((small, small), np.uint8)
    kernel[int(small / 2)] = np.ones(small)
    # Dilate mask
    mask_decimated = cv2.dilate(mask, kernel, iterations = 1)
    
    # DELETE ME
    #cv2.imwrite("mask_Horiz_dilated.jpg", mask_decimated)
    
    # Open mask iterations times to clean/combine inital contours
    mask_decimated = cv2.morphologyEx(mask_decimated, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(small,small)), iterations = iterations)
    
    
    # DELETE ME
    #cv2.imwrite("mask_decimated.jpg", mask_decimated)
    
    
    # Find contour data from decimated mask
    cont_full, hierarchy = cv2.findContours(mask_decimated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    
    # DELETE ME
    print("All contours:", len(cont_full))
    
    
    contBounds = []
    contCenters = []
    contArea = []
    crop_images = []

    # Keep contours that survived the decimation, but keep pre-decimated shape
    for i,c in enumerate(cont_full):
        m = cv2.moments(c)
        
        center = (int(m['m10']/m['m00']) if m['m00'] != 0 else 0, int(m['m01']/m['m00']) if m['m00'] else 0)
        
        if mask_decimated[center[1]][center[0]] != 0:
            contBounds.append(cv2.boundingRect(c))
            contCenters.append(center)
            contArea.append(m['m00'])


    # DELETE ME
    print("Kept contours:", len(contBounds))
    
    if len(contBounds) > 160:
        raise Exception()
    
    #imgBoundaries = image.copy()
    cutBounds = []
    
    for i,c in enumerate(contBounds):
        # Split contour bounding data
        x, y, w, h = c
        
        # Expand contour bounds to include potential waldo heads, recalculate bounds
        xEx = int(x - (w * scale_x - w) / 2)
        yEx = int(y - (h * scale_y - h) / 2)
        wEx = int(xEx + w * scale_x)
        hEx = int(yEx + h * scale_y)
        # Set final outer bounds safely within image boundary
        xTop = xEx if xEx in range(imShape[1]) else (0 if xEx < 0 else imShape[1])
        yTop = yEx if yEx in range(imShape[0]) else (0 if yEx < 0 else imShape[0])
        xBot = wEx if wEx in range(imShape[1]) else (0 if wEx < 0 else imShape[1])
        yBot = hEx if hEx in range(imShape[0]) else (0 if hEx < 0 else imShape[0])
        
        # Crop bounded contour out from original image
        crop_images.append(image[yTop:yBot, xTop:xBot].copy())
        
        cutBounds.append([(xTop,yTop), (xBot,yBot)])
        
        # Base bounding in green
        #cv2.rectangle(imgBoundaries, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Contour center in blue
        #cv2.circle(imgBoundaries, contCenters[i], 2, (255,0,0), 2)
        # Expanded bounding in red
        #cv2.rectangle(imgBoundaries, (xEx, yEx), (wEx, hEx), (0, 0, 255), 2)
        
        #cv2.imwrite("Cutouts/CropTest_" + str(i) + ".jpg", crop_images[i])

    #cv2.imwrite("ContourBounds.jpg", imgBoundaries)
    
    return crop_images, cutBounds

def FindWaldo(imname):
    # Open original image as byte image for Contour detecting
    original = img_as_ubyte(io.imread(imname ,as_gray=True))
    # Convert to HSV color mode
    im_HSV = cv2.cvtColor(cv2.imread(imname), cv2.COLOR_BGR2HSV)
    
    # Bounds for red and white in image in HSV
    lo1_r = (0, 102, 102)
    hi1_r = (9, 255, 255)
    
    lo2_r = (171, 102, 102)
    hi2_r = (180, 255, 255)
    
    lo_w = (0, 0, 229)
    hi_w = (180, 51, 255)
    
    # Get red and white masks using color bounds
    mask_r = cv2.inRange(im_HSV, lo1_r, hi1_r) | cv2.inRange(im_HSV, lo2_r, hi2_r)
    mask_w = cv2.inRange(im_HSV, lo_w, hi_w)
    
    # Construct vertical kernel for dilating vertically using kernel size 1% of the images height
    kernelSize = int(.01 * original.shape[0]) if int(.01 * original.shape[0] % 2) == 1 else int(.01 * original.shape[0]) - 1
    kernelSize = kernelSize if kernelSize > 2 else 3
    
    kernel = np.zeros((kernelSize, kernelSize), np.uint8)
    kernel[int(kernelSize / 2)] = np.ones(kernelSize)
    kernel = np.transpose(kernel)
    
    # Dilate red and white masks vertically
    mask_r_dilated = cv2.dilate(mask_r, kernel, iterations = 1)
    mask_w_dilated = cv2.dilate(mask_w, kernel, iterations = 1)
    # Combine red and white masks by bitwise AND
    mask = mask_r_dilated & mask_w_dilated
    
    #orig = cv2.imread(imname)
    #result = cv2.bitwise_and(orig, orig, mask = mask)
    
    # Build contour array
    cutouts, cutBounds = ContourBounds(original, mask, (original.shape[0], original.shape[1]), kernelSize)
    
    img_out = io.imread(imname)
    # Walk cutout contoured objects and check for circles
    for i,c in enumerate(cutouts):
        if DetectCircles(img_as_ubyte(c), "cutout_" + str(i)):
            cv2.rectangle(img_out, cutBounds[i][0], cutBounds[i][1], (0, 0, 255), 5)
            
            
    io.imsave("Final_out.jpg", img_out)
        
    # MASK OUTPUT
    cv2.imwrite("mask_red.jpg", mask_r)
    cv2.imwrite("mask_white.jpg", mask_w)
    
    cv2.imwrite("mask_red_dilated.jpg", mask_r_dilated)
    cv2.imwrite("mask_white_dilated.jpg", mask_w_dilated)
    
    cv2.imwrite("mask_Final.jpg", mask)
    #cv2.imwrite("Image_Masked.jpg", result)

    
FindWaldo("images/hard.jpg")