import glob
import numpy as np
import cv2
import math
from skimage.feature import peak_local_max


def linearHoughTransform(img, edges):
    imcpy = img.copy()
    lines = cv2.HoughLinesP(edges, cv2.HOUGH_PROBABILISTIC, np.pi / 180, 45, minLineLength=5, maxLineGap=150)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        pts = np.array([[x1, y1], [x2, y2]], np.int32)
        cv2.polylines(imcpy, [pts], True, (0, 255, 0), 3)
    return imcpy


def parabolicHoughTransform(img, centrox, centroy, pmin, pmax, edges):
    imcpy = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vector_p = np.linspace(-pmax, pmax)
    vector_phi = np.linspace(0, 2 * np.pi - (2 * np.pi / 100))
    Accumulator = np.zeros([len(vector_phi), len(vector_p)])
    y = img_gray[0]
    x = img_gray[1]
    for i in range(0, len(x)):
        for j in range(0, len(vector_phi)):
            Y = int(y[i]) - int(centroy[i])
            X = int(x[i]) - int(centrox[i])
            angulo = vector_phi[j]
            numerador = pow((Y * math.cos(angulo) - X * math.sin(angulo)), 2)
            denominador = 4 * (X * math.cos(angulo) + Y * math.sin(angulo))
            if denominador != 0:
                p = numerador / denominador
                if pmin < abs(p) < pmax and p != 0:
                    indice = np.argwhere(vector_p >= p)
                    indice = indice[0]
                    Accumulator[j][indice] += 1
    maximo = peak_local_max(Accumulator)
    idx_phi, idx_p = maximo[0]
    p = vector_p[idx_p]
    phi = vector_phi[idx_phi];
    lines = cv2.HoughLinesP(edges, cv2.HOUGH_PROBABILISTIC, np.pi / 180, int(p), minLineLength=0, maxLineGap=int(phi))
    for line in lines:
        x1, y1, x2, y2 = line[0]
        pts = np.array([[x1, y1], [x2, y2]], np.int32)
        cv2.polylines(imcpy, [pts], True, (0, 0, 255), 3)
    return imcpy


def threshold(img, low, high):
    # Detecting lower and upper bounds for threshold
    lowerBound = np.array(low, dtype="uint8")
    upperBound = np.array(high, dtype="uint8")

    # Applying the threshold and return the inverse of it as we need to background to be black
    thresh = cv2.inRange(img, lowerBound, upperBound)
    return cv2.bitwise_not(thresh)


def main():
    for i, filename in enumerate(glob.glob('Dataset/*.bmp')):
        # Reading all the images from Dataset folder
        img = cv2.imread(filename)

        # Applying the threshold for RGB images
        thresh = threshold(img, 100, 255)

        # Applying Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

        # Applying Canny edge detection
        edges = cv2.Canny(opening, 100, 200, apertureSize=3)

        limg = linearHoughTransform(img, edges)
        pimg = parabolicHoughTransform(img, edges[0], edges[1], 10, 20, edges)

        disp_h = np.hstack((limg, pimg))

        cv2.imshow('Image {}'.format(i), disp_h)

    cv2.waitKey()


if __name__ == "__main__":
    main()
