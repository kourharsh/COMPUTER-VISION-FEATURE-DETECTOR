import cv2
import numpy as np

window = 5
padding = np.uint8((window-1)/2)  # == 2

# for local_maximum
max_window = 3
max_padding = np.uint8((max_window-1)/2)  # == 1
threshold = 50000000

def main():
    image1 = cv2.imread("image_sets/yosemite/Yosemite1.jpg", 3)
    image2 = cv2.imread("image_sets/yosemite/Yosemite2.jpg", 3)

    gray_img1, sobel_x1, sobel_y1 = calcimagederivate(image1)
    gray_img2, sobel_x2, sobel_y2 = calcimagederivate(image2)

    points1, point_count1, strength_mat1 = calcinterestpoints(sobel_x1, sobel_y1, gray_img1)
    points2, point_count2, strength_mat2 = calcinterestpoints(sobel_x2, sobel_y2, gray_img2)

    max_strength_mat1, max_points1, max_point_count1 = calclocalmax(strength_mat1)
    max_strength_mat2, max_points2, max_point_count2 = calclocalmax(strength_mat2)

    cv2.imshow('Yosemite1', image1)
    cv2.imshow('Yosemite2', image2)
    key = cv2.waitKey(0)
    if key == 13:  # waiting for enter key to exit
        cv2.destroyAllWindows()
        cv2.waitKey(1)

def calclocalmax(strength_mat):
    height = strength_mat.shape[0]
    width = strength_mat.shape[1]
    k = 1

    print("max window: " + str(max_window))
    print("max padding: " + str(max_padding))
    for y in range(0, height-max_window):
        for x in range(0, width-max_window):
            mat = submat(strength_mat, y, x, max_window)
            max = np.amax(mat)
            if max != 0:
                if k==1:
                    #print(max_index[0])
                    #print(max_index[1])
                    k=0

                if strength_mat[y+max_padding, x+max_padding] == max:
                    strength_mat[y:y + max_window, x:x + max_window] = 0
                    strength_mat[y+max_padding, x+max_padding] = max


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
    #print(height)
    #print(width)

    point_count = 0

    points = []

    strength_mat = np.zeros(image.shape, np.uint)

    k=1

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

            corner_strength = np.uint(determinant / trace)
            if corner_strength > threshold:
                strength_mat[y+padding, x+padding] = corner_strength
                point_count = point_count + 1
                points.append([y+padding, x+padding])

            #if k == 1:
            #    print(num_xx)
            #    print(num_xy)
            #    print(num_yy)
            #    k=0

    #print(points)
    #print(point_count)
    print("Interest Points: " + str(point_count))

    return points, point_count, strength_mat


def calcimagederivate(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)
    #sobel_x = np.absolute(sobel_x) # taking absolute values to not loose negative values of gradient
    #sobel_y = np.absolute(sobel_y)
    #sobel_x = np.uint8(sobel_x)
    #sobel_y = np.uint8(sobel_y)
    #print(sobel_x)
    #print(sobel_y)

    return gray_img, sobel_x, sobel_y

if __name__ == "__main__":
    main()