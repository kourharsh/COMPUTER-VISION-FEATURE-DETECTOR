*****README FILE*****

PREREQUISITES:-

Programming Language:- Python3

Libraries/Packages:-
	cv2 
	numpy


Required files:-
1. Input first .ppm or.jpg or.png image file 
2. Input second .ppm or.jpg or.png image file 


RUN FILE:-
1. Put both the input files in the image_set folder.
2. Run A2.py
3. Enter the full location of the first file with extension eg: "image_sets/graf/img1.ppm"
4. Enter the full location of the second file with extension eg: "image_sets/graf/img2.ppm"
5. An output image displaying the feature keypoint matches from both the images will be displayed.
6. Press ENTER KEY by selecting the displayed image to exit the program and close all the displayed images.


Functionality implemented:
I have implemented the following major functionalities:
	
A. def calcimagederivate(image): 1. To convert the image into grayscale and calculate the image derivative.


B. def calcinterestpoints(sobelx , sobely, image): 1. Calculate the Harris matrix as per the neighbourhood of [5x5]
						2. Compute the Corner Strength
						3. Apply the Threshold
						4. Returns a list of interest point above the threshold.

C. def calclocalmax(strength_mat, threshold): 1. Check if the interest point achieved is the local maxima in neighbourhood of [3x3].
					   2. If the interest point is a local maxima in the neighbourhood, keep the point else remove it and replace the response to zero. 


D. def adaptive_nonmax_suppression(max_strength_mat, max_points, max_point_count): 1. Compute Adaptive non-maximal suppression.
				2. This ensures the features to be scattered in the image. It works by computing the suppression radius for every key point and sort the key points based on radius computed in decreasing order and picked top 500 keypoints.


E. def constructsiftdescriptor(gray_img, adaptive_points): 1. Calculate the magnitude and angle of entire image.
						2. For each key point, take a window of [16x16] around the key point and normalise the magnitude.
						3. Divide window into 16 * [4x4] windows. Compute the rotation invariance by dividing the orientation of the window into 36 bins of 10 degrees each. I have used the voting mechanism by adding the magnitude to the vote.For each bin having the vote above the 80% of dominated orientation, rotate the window and create a separate descriptor.
						4. Creating the descriptor I have clipped the histograms to 0.2 in order to achieve contrast invariance and then normalized the 128-dimensional descriptor.


F. def featurematching(keyp1,keyp2): 1. For every key point find the best match according to threshold SSD distance.
				  2. For key points having multiple descriptors because of rotational invariance, choose the descriptor with minimum distance.
				  3. Applied SSD ratio test to avoid the ambiguous matches.



References:
1. http://cs.brown.edu/courses/csci1950-g/results/proj6/steveg/theory.html
2. http://www.scholarpedia.org/article/Scale_Invariant_Feature_Transform 
3. https://www.researchgate.net/publication/323388062_Efficient_adaptive_non-maximal_suppression_algorithms_for_homogeneous_spatial_keypoint_distribution





 

