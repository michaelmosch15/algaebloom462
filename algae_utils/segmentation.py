import cv2
import numpy as np

def preprocess(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    return lab

def otsu_threshold(lab_img):
    _, mask = cv2.threshold(lab_img[:,:,1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def kmeans_segmentation(image, k=3):
    Z = image.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    label = label.flatten()
    centers_lab = cv2.cvtColor(center.astype(np.uint8).reshape(1,-1,3), cv2.COLOR_BGR2LAB)[0]
    algae_cluster = np.argmax(centers_lab[:,1])
    mask = (label==algae_cluster).astype(np.uint8)*255
    mask = mask.reshape(image.shape[:2])
    return mask

def clean_mask(mask, kernel_size=5, iterations=2):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return mask