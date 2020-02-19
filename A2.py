import cv2
import numpy as np
import math

window = 5
padding = np.uint8((window-1)/2)  # == 2

# for local_maximum
max_window = 3
max_padding = np.uint8((max_window-1)/2)  # == 1

threshold = 20000000

def main():
    #image1 = cv2.imread("image_sets/graf/img1.ppm", 3)
    #image2 = cv2.imread("image_sets/graf/img2.ppm", 3)
    #image1 = cv2.imread("image_sets/yosemite/Yosemite1.jpg", 3)
    #image2 = cv2.imread("image_sets/yosemite/Yosemite2.jpg", 3)
    #image1 = cv2.imread("image_sets/pano/pano1_0008.png", 3)
    #image2 = cv2.imread("image_sets/pano/pano1_0009.png", 3)

    img1 = input("Enter the image 1 address : ")
    img2 = input("Enter the image 2 address : ")
    image1 = cv2.imread(img1, 3)
    image2 = cv2.imread(img2, 3)


    gray_img1, sobel_x1, sobel_y1 = calcimagederivate(image1)
    gray_img2, sobel_x2, sobel_y2 = calcimagederivate(image2)

    points1, point_count1, strength_mat1, threshold1 = calcinterestpoints(sobel_x1, sobel_y1, gray_img1)
    points2, point_count2, strength_mat2, threshold2 = calcinterestpoints(sobel_x2, sobel_y2, gray_img2)

    max_strength_mat1, max_points1, max_point_count1 = calclocalmax(strength_mat1, threshold1)
    max_strength_mat2, max_points2, max_point_count2 = calclocalmax(strength_mat2, threshold2)

    adaptive_points1 , pixel_count1 = adaptive_nonmax_suppression(max_strength_mat1, max_points1, max_point_count1)
    adaptive_points2 , pixel_count2 = adaptive_nonmax_suppression(max_strength_mat2, max_points2, max_point_count2)

    desc_points1, desc_keypoints1 = constructsiftdescriptor(gray_img1, adaptive_points1)
    desc_points2, desc_keypoints2 = constructsiftdescriptor(gray_img2, adaptive_points2)

    matched_dict, distanceratiodict, finalpoints, matchedcount = featurematching(desc_points1,desc_points2)

    Matches = cv2.drawMatches(image1, desc_keypoints1, image2, desc_keypoints2, finalpoints, None)


    cv2.imshow('Matches', Matches)
    key = cv2.waitKey(0)
    if key == 13:  # waiting for enter key to exit
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def featurematching(keyp1,keyp2):
    second_occupied = []
    keypointdict = {}
    finalpoints= []
    matched_dict = {}
    matched_dict_dist = {}
    matched_dict_second_dist = {}
    distanceratiodict = {}
    out_index = 0
    for i_o in range(0, len(keyp1)):
        y1, x1, dist1 = keyp1[i_o]
        mindist  = 100
        secondmindist =100
        keypoints = []
        index = 0
        for i_i in range(0, len(keyp2)):
            y2, x2, dist2 = keyp2[i_i]
            dist = (dist1 - dist2) ** 2
            dist = dist.sum()
            if dist < mindist:
                secondmindist = mindist
                mindist = dist
                keypoints.append([y1, x1])
                keypoints.append([y2, x2])
                index = i_i

        if mindist < 0.3: #threshold match -SSD distance
            s = str(y1) + "," + str(x1)
            if s in matched_dict_dist.keys():
                d = matched_dict_dist[s]
                if mindist < d:
                    matched_dict[s] = [i_o , index]
                    matched_dict_dist[s] = mindist
                    matched_dict_second_dist[s] = secondmindist
                    keypointdict[s] = keypoints
            else:
                matched_dict[s] = [i_o, index]
                matched_dict_dist[s] = mindist
                matched_dict_second_dist[s] = secondmindist
                keypointdict[s] = keypoints

    for key in matched_dict.keys():
        ratio = matched_dict_dist[key]/matched_dict_second_dist[key]
        if ratio < 0.5: ## SSD ratio
            idx = matched_dict[key]
            outer_index = idx[0]
            inner_index = idx[1]
            if inner_index not in second_occupied:
                finalpoints.append(cv2.DMatch(outer_index, inner_index, matched_dict_dist[key]))
                second_occupied.append(inner_index)


    matchedcount = len(finalpoints)
    print("Matched Keyspoints : " + str(matchedcount))
    print(finalpoints)

    return matched_dict, distanceratiodict, finalpoints, matchedcount


def get_magntheta(h, w, gray_img):
    magnitude = np.zeros(gray_img.shape, np.float)
    theta = np.zeros(gray_img.shape, np.float)
    for y in range(1, h):
        for x in range(1, w):
            Lx1_y = int(gray_img[y, x+1]) - int(gray_img[y,x-1])
            Lx_y1 = int(gray_img[y+1, x]) - int(gray_img[y-1,x])
            magnitude[y,x] = ((Lx1_y ** 2) + (Lx_y1 ** 2))** 0.5
            val = np.arctan2(Lx_y1, Lx1_y)
            theta[y, x] = np.degrees(val)
    return magnitude, theta

def constructsiftdescriptor(gray_img, adaptive_points):
    desc_points = []
    h = gray_img.shape[0]-1
    w = gray_img.shape[1]-1
    magnitude, theta = get_magntheta(h, w, gray_img)

    k=1
    #print(theta)
    for y, x,d in adaptive_points:

        if y-8 >= 0 and y+8 < gray_img.shape[0] and x-8 >= 0 and x+8 < gray_img.shape[1]:

            mag_mat = submat(magnitude, y-8, x-8, 16)
            theta_mat = submat(theta, y - 8, x - 8, 16)

            mag_mat = cv2.normalize(mag_mat, None, norm_type=cv2.NORM_L2)

            theta_final = rotation_n(theta_mat, mag_mat, k)  # rotation invariance
            for index in range(0, len(theta_final)):
                sift_desc = histogram_main(mag_mat, theta_final[index])
                desc_points.append([y, x, sift_desc])

    print("sift descriptor : " + str(len(desc_points)))
    desc_keypoints = []
    for y, x, z in desc_points:
        desc_keypoints.append(cv2.KeyPoint(x, y, 1))

    return desc_points, desc_keypoints

def histogram_main(mag_mat, theta_mat):
    sift_desc = []
    indexes = [0, 4, 8, 12]
    # divide 16*16 mat into 8 4*4 matrix
    for i in indexes:
        for j in indexes:
            sub_theta = submat(theta_mat, i, j, 4)
            sub_mag = submat(mag_mat, i, j, 4)
            # calculatehistogram
            voted_hist = createhistogram(sub_mag, sub_theta)
            sift_desc.append(voted_hist)
    sift_desc = np.array(sift_desc).reshape(-1)
    sift_desc = cv2.normalize(sift_desc, None, norm_type=cv2.NORM_L2)
    return sift_desc


def rotation_n(theta_mat,mag_mat,k):
    theta_final = []
    bin = []
    list_r = []
    for i in range(0 , 36):
        list_r.append(0)
    for l in range(0, 16):
        for m in range(0, 16):
            angle = theta_mat[l][m]
            if angle < 0:
                angle = angle + 360
            elif angle > 360:
                angle = angle % 360
            theta_mat[l][m] = angle
            val = int(math.floor(angle / 10))
            mag = mag_mat[l][m] + list_r[val]
            list_r[val] = mag

    mtheta = max(list_r)
    max_theta = 0.8 * mtheta


    for j in range(0, len(list_r)): #get indexes of bins with val > max_theta
        if list_r[j] >= max_theta:
            bin.append(j)

    for p in range(0, len(bin)):
        angle_r = (bin[p]) * 10
        theta_new = rotatewindow(theta_mat, angle_r,k)
        theta_final.append(theta_new)

    return theta_final


def rotatewindow(theta_mat, binval,k):
    for l in range(0, 16):
        for m in range(0, 16):
            val = theta_mat[l][m] - binval
            if val < 0:
                val = val + 360
            elif val > 360:
                val = val % 360
            theta_mat[l][m] = val
    return theta_mat

def createhistogram(sub_mag,sub_theta):
    dict = {}
    for y in range(0,4):
        for x in range(0,4):
            angle = sub_theta[y, x]
            if angle < 0:
                angle = angle + 360
            elif angle > 360:
                angle = angle % 360
            val = int(math.floor(angle / 45))
            if val in dict.keys():
                mag = dict[val] + sub_mag[y,x]
                dict[val] = mag
            else:
                dict[val] = sub_mag[y,x]

    voted_hist = []
    for key in range(0,8):
        if key in dict :
            if dict[key] < 0.2: #contrast invariance or normalizing the value of histograms

                voted_hist.append(dict[key])
            else:
                voted_hist.append(0.2)
        else:
            voted_hist.append(0.0)


    return voted_hist


def sortdistance(val):
    return val[2]

def adaptive_nonmax_suppression(max_strength_mat, max_points, max_point_count):
    adaptive_points = []
    print("Performing adaptive_nonmax_suppression")
    #k=1
    for y_o, x_o in max_points:
        min_distance = 999999999999999999
        strength_o = max_strength_mat[y_o][x_o]
        for y_i, x_i in max_points:
            strength_i = max_strength_mat[y_i][x_i]
            if y_i != y_o and x_i != x_o and(strength_o < (0.9 * strength_i)):
                distance = (((y_o - y_i) ** 2) + ((x_o - x_i) ** 2)) ** (1.0 / 2)  # eucledian distance
                if distance <= min_distance:
                    min_distance = distance
        adaptive_points.append([y_o,x_o,min_distance])

    adaptive_points.sort(key = lambda x: x[2], reverse= True)

    #pixel_count = np.uint(.9 * (len(adaptive_points))) #60% of the pixels #to be changed
    pixel_count = 500
    print("Adaptive points : " )
    print(adaptive_points[0:pixel_count])
    return adaptive_points[0:pixel_count], pixel_count

def calclocalmax(strength_mat, threshold):
    height = strength_mat.shape[0]
    width = strength_mat.shape[1]
    print("max window: " + str(max_window))
    print("max padding: " + str(max_padding))
    for y in range(0, height- max_window):
        for x in range(0, width - max_window):
            mat = submat(strength_mat, y, x, max_window)
            minval, maxval, minloc, maxloc = cv2.minMaxLoc(mat)
            if maxval > threshold:
                temp = np.zeros((max_window, max_window), np.float32)
                temp[maxloc] = maxval
                strength_mat[y:y + max_window, x:x + max_window] = temp

    max_points = np.transpose(np.nonzero(strength_mat)) #Co-ordinates with strength > local maximum
    max_point_count = np.count_nonzero(strength_mat)
    #print(max_points)
    print("Points after local maximum: " + str(max_point_count))
    return strength_mat, max_points, max_point_count

def submat(mat, startRow, startCol, size):
    return mat[startRow:startRow+size, startCol:startCol+size]


def calcinterestpoints(sobelx , sobely, image):
    Ixx = sobelx * sobelx
    Iyy = sobely * sobely
    Ixy = sobelx * sobely

    blur_xx = cv2.GaussianBlur(Ixx, (3, 3), 1) #window function
    blur_yy = cv2.GaussianBlur(Iyy, (3, 3), 1)
    blur_xy = cv2.GaussianBlur(Ixy, (3, 3), 1)

    height = image.shape[0]
    width = image.shape[1]

    point_count = 0

    points = []

    strength_mat = np.zeros(image.shape, np.uint)

    for y in range(0, height-window):
        for x in range(0, width-window):

            num_xx = submat(blur_xx, y, x, window)
            num_yy = submat(blur_yy, y, x, window)
            num_xy = submat(blur_xy, y, x, window)

            sum_xx = num_xx.sum()
            sum_yy = num_yy.sum()
            sum_xy = num_xy.sum()

            trace = sum_xx + sum_yy
            determinant = (sum_xx * sum_yy) - (sum_xy * sum_xy)
            if trace == 0:
                trace = 0.9
            corner_strength = determinant / trace
            if np.isnan(corner_strength):
                corner_strength = 0
            else:
                corner_strength = np.uint(corner_strength)
            strength_mat[y+padding, x+padding] = corner_strength

    #golobal_minval, golobal_maxval, golobal_minloc, golobal_maxloc = cv2.minMaxLoc(strength_mat)
    #print(golobal_maxval)
    #threshold = golobal_maxval * .2  #to be changed
    #threshold = 20000000
    print(threshold)

    for y in range(0, height):
        for x in range(0, width):
            strength = strength_mat[y, x]
            if strength > threshold:
                strength_mat[y, x] = strength
                point_count = point_count + 1
                points.append([y, x])
            else:
                strength_mat[y, x] = 0
    print("Interest Points: " + str(point_count))

    return points, point_count, strength_mat, threshold


def calcimagederivate(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)
    return gray_img, sobel_x, sobel_y

if __name__ == "__main__":
    main()