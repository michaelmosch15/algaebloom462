import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from algae_utils.segmentation import preprocess, kmeans_segmentation, clean_mask
from algae_utils.features import extract_features
from algae_utils.classification import build_dnn
import os

def analyze_lake():
    tiles = []
    tile_labels = []  
    water_data_dir = "water_data" 

    for i in range(3):
        for j in range(3):
            fname = os.path.join(water_data_dir, f'water_tile_{i}_{j}.jpg')
            img = cv2.imread(fname)
            if img is None:
                print(f"this tile is missing -- {fname}")
                continue
            tiles.append(img)
            tile_labels.append(1) 

    all_features = []
    all_labels = []
    for idx, tile in enumerate(tiles):
        lab = preprocess(tile)
        mask = kmeans_segmentation(tile)
        mask = clean_mask(mask)
        features = extract_features(tile, mask)
        if features.shape[0] > 0:
            all_features.append(features)
            all_labels.extend([tile_labels[idx]] * features.shape[0])

    if len(all_features) == 0:
        return None

    X = np.vstack(all_features)
    y = np.array(all_labels)
    y_cat = to_categorical(y, 2)

    model = build_dnn(X.shape[1])
    model.fit(X, y_cat, epochs=20, batch_size=8, verbose=1)

    full_img_path = os.path.join(water_data_dir, 'water_full.jpg')
    full_img = cv2.imread(full_img_path)
    if full_img is None:
        print(f"Full lake image ({full_img_path}) is not there")
        return None

    # lake edge detection/contours/stuff like that
    gray_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, threshold1=50, threshold2=150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lake_mask = np.zeros_like(gray_img, dtype=np.uint8)
    cv2.drawContours(lake_mask, contours, -1, (255), thickness=cv2.FILLED)

    # get rid of woods and stuff and black areas
    hsv_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv_img, (35, 40, 40), (85, 255, 255))  
    lake_mask = cv2.bitwise_and(lake_mask, cv2.bitwise_not(green_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    lake_mask = cv2.dilate(lake_mask, kernel, iterations=2)

    non_black_mask = (gray_img > 10).astype(np.uint8) * 255
    lake_mask = cv2.bitwise_and(lake_mask, non_black_mask)

    lab_full = preprocess(full_img)
    mask_full = kmeans_segmentation(full_img)
    mask_full = clean_mask(mask_full)

    mask_full = cv2.bitwise_and(mask_full, mask_full, mask=lake_mask)

    features_full = extract_features(full_img, mask_full)

    if features_full.shape[0] > 0:
        preds = model.predict(features_full)
        pred_classes = np.argmax(preds, axis=1)

        labeled_mask = cv2.connectedComponents(mask_full)[1]


        algae_mask_pred = np.zeros_like(mask_full)
        region_vals = np.unique(labeled_mask)[1:]
        for i, region_label in enumerate(region_vals):
            if i < len(pred_classes) and pred_classes[i] == 1:
                algae_mask_pred[labeled_mask == region_label] = 255

        algae_mask_pred = cv2.bitwise_and(algae_mask_pred, algae_mask_pred, mask=lake_mask)


        lake_area = np.count_nonzero(lake_mask)
        algae_area = np.count_nonzero(algae_mask_pred)


        overlay = full_img.copy()
        algae_overlay = cv2.addWeighted(overlay, 0.7, cv2.cvtColor(algae_mask_pred, cv2.COLOR_GRAY2BGR), 0.3, 0)
        
        # Return both the overlay and the area scores
        return algae_overlay, algae_area, lake_area

    else:
        print("No contamination regions found")
        return None