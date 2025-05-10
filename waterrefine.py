from wateranalysis import analyze_lake
import cv2
import numpy as np

def create_lake_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_lake = np.array([80, 20, 10])
    upper_lake = np.array([140, 120, 105])
    mask = cv2.inRange(hsv, lower_lake, upper_lake)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    lake_mask = np.zeros_like(mask)
    cv2.drawContours(lake_mask, [largest], -1, 255, cv2.FILLED)
    return lake_mask

def create_algae_mask(image, lake_mask):

    lake_only = cv2.bitwise_and(image, image, mask=lake_mask)
    hsv = cv2.cvtColor(lake_only, cv2.COLOR_BGR2HSV)


    lower_algae = np.array([0, 0, 80])
    upper_algae = np.array([180, 70, 255])
    algae_mask = cv2.inRange(hsv, lower_algae, upper_algae)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    algae_mask = cv2.morphologyEx(algae_mask, cv2.MORPH_OPEN, kernel)

    algae_mask = cv2.bitwise_and(algae_mask, algae_mask, mask=lake_mask)
    return algae_mask

def refined_water_data():
    algae_overlay, _, _ = analyze_lake()
    if algae_overlay is None:
        return None, 0, 0

    lake_mask = create_lake_mask(algae_overlay)
    if lake_mask is None or lake_mask.sum() == 0:
        return None, 0, 0
    refined_lake_area = np.count_nonzero(lake_mask)

    algae_mask = create_algae_mask(algae_overlay, lake_mask)
    refined_algae_area = np.count_nonzero(algae_mask)

    refined_algae_overlay = algae_overlay.copy()
    contours, _ = cv2.findContours(algae_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(refined_algae_overlay, contours, -1, (0,0,255), 2)

    return refined_algae_overlay, refined_algae_area, refined_lake_area