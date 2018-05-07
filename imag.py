import cv2
import numpy as np
from matplotlib import pyplot as plt

fn = "01_h"  # file name
data = cv2.imread("./all/images/" + fn + ".jpg", 0)
color = cv2.imread("./all/images/" + fn + ".jpg", 1)
manual = cv2.imread("./all/manual1/" + fn + ".tif", 0)
mask = cv2.imread("./all/mask/" + fn + "_mask.tif", 0)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

scale = 0.25

data = cv2.resize(data, None, fx=scale, fy=scale)

# sharpen - not used
kernelsh = np.zeros((9, 9), np.float32)
kernelsh[4, 4] = 2.0
boxFilter = np.ones((9, 9), np.float32) / 81.0
kernelsh = kernelsh - boxFilter
sharpen = cv2.filter2D(data, -1, kernelsh)

# equalize histogram - not used
# datah = cv2.equalizeHist(data)

# clahe - not used
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# datac = clahe.apply(data)

# przetwarzanie obrazu
manual = cv2.resize(manual, None, fx=scale, fy=scale)
mask = cv2.resize(mask, None, fx=scale, fy=scale)
color = cv2.resize(color, None, fx=scale, fy=scale)

datag = cv2.GaussianBlur(data, (3, 3), 0)
thresh = cv2.adaptiveThreshold(datag, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
gaussed = cv2.GaussianBlur(thresh, (5, 5), 0)
canny = cv2.Canny(data, 20, 130)
opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
opened = opened * (mask / 255.0)

# cv2.imwrite("./output/" + fn + "out.png", opened)
cv2.imwrite(fn + ".jpg", opened)
color = color[..., ::-1]  # konflikt cv2 z plt

for row in range(len(color)):
    for col in range(len(color[row])):
        if opened[row][col] == 255:
            color[row][col] = [255, 255, 255]

titles = ['original', 'thresh', 'gaussed', 'opened', 'manual']
images = [data, thresh, gaussed, opened, manual]

for i in range(len(titles)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])

plt.subplot(2, 3, 6)
plt.imshow(color)
plt.title("output")

color = color[..., ::-1]
cv2.imwrite(fn + "col.jpg", color)

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
