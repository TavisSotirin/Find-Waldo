# Find-Waldo
Python script to find Waldo in 'Where's Waldo' pictures using image processing techniques

ABSTRACT
The purpose of this script is to be capable of searching for and identifying potential “Waldo’s” in a given image from a “Where’s Waldo” book. 
Written in Python using OpenCV, Scikit-Image, Numpy, and the built in Math libraries. 

The code is run as a script with a manually entered input image. It operates on files in the local directory.
The program converts the image to HSV color mode and segments it based on red and white, creating binary masks for 
each. The masks are dilated vertically and the overlap between them is kept attempting to identify horizontally striped 
red and white objects (such as Waldo’s shirt). The resultant mask is horizontally dilated to combine nearby objects and 
then opened several times to remove any noise. The final mask is bounded by its contours and the bounds are then 
expanded and cropped from the original image. Ideally, one of these cropped images contains Waldo. At this point the 
cropped images are individually scaled up, 3-threshold grey banded, sharpened, and have edges extracted by the Canny 
edge operator. The edges are then run through a circle Hough transform to identify any circles. The circles that are 
horizontal from one another and within a distance threshold apart are kept and returned. Any cropped image that 
returns at least 2 circles is considered a potential Waldo and highlighted in the original image. The highlighted original 
image is then returned.
