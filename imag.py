import cv2
import numpy as np
from matplotlib import pyplot as plt

fn = "07_g"
data = cv2.imread("./all/images/" + fn + ".jpg", 0)
manual = cv2.imread("./all/manual1/" + fn + ".tif", 0)
mask = cv2.imread("./all/mask/" + fn + "_mask.tif", 0)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

scale = 0.25

data = cv2.resize(data, None, fx=scale, fy=scale)
manual = cv2.resize(manual, None, fx=scale, fy=scale)
mask = cv2.resize(mask, None, fx=scale, fy=scale)
thresh = cv2.adaptiveThreshold(data, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
gaussed = cv2.GaussianBlur(thresh, (5, 5), 0)
canny = cv2.Canny(data, 20, 130)
opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
opened = opened * (mask/255.0)

titles = ['original', 'thresh', 'gaussed', 'opened', 'canny', 'manual']
images = [data, thresh, gaussed, opened, canny, manual]

for i in range(len(titles)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])

tp = 0
tn = 0
fp = 0
fn = 0
total = 0

siz = np.array(manual).shape
print(siz)
size = siz[0] * siz[1]

for row in range(len(manual)):
    for col in range(len(manual[row])):
        if manual[row][col] == 255:
            if opened[row][col] == 255:
                tp += 1
            else:
                fn += 1
        else:
            if opened[row][col] == 255:
                fp += 1
            else:
                tn += 1

print("TP: " + str(tp))
print("FP: " + str(fp))
print("FN: " + str(fn))
print("TN: " + str(tn))

print('Sensitivity: ' + str(float(tp) / (tp + fn)))
print('Precision: ' + str(float(tp) / (tp + fp)))
print('Accuracy: ' + str(float(tp + tn) / (tp + tn + fp + fn)))

plt.show()
