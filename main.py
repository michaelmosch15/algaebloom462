from datapull import main as datapull_main
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
                print(f"Tile image missing: {fname}, skipping.")
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
        print("Feature extraction error")
        return

    X = np.vstack(all_features)
    y = np.array(all_labels)
    y_cat = to_categorical(y, 2)

    model = build_dnn(X.shape[1])
    model.fit(X, y_cat, epochs=20, batch_size=8, verbose=1)

    full_img_path = os.path.join(water_data_dir, 'water_full.jpg')
    full_img = cv2.imread(full_img_path)
    if full_img is None:
        print(f"Full lake image ({full_img_path}) is not there")
        return
    lab_full = preprocess(full_img)
    mask_full = kmeans_segmentation(full_img)
    mask_full = clean_mask(mask_full)

    features_full = extract_features(full_img, mask_full)

    if features_full.shape[0] > 0:
        preds = model.predict(features_full)
        pred_classes = np.argmax(preds, axis=1)

        labeled_mask = cv2.connectedComponents(mask_full)[1]
        algae_mask_pred = np.zeros_like(mask_full)
        region_vals = np.unique(labeled_mask)[1:]  # skip background 0
        for i, region_label in enumerate(region_vals):
            if i < len(pred_classes) and pred_classes[i] == 1:
                algae_mask_pred[labeled_mask == region_label] = 255

        lake_area_mask = np.ones(full_img.shape[:2], dtype=np.uint8)
        lake_area = np.count_nonzero(lake_area_mask)
        algae_area = np.count_nonzero(algae_mask_pred)
        contamination_score = algae_area / lake_area * 100
        print(f"Lake contamination score: {contamination_score:.2f}%")

        overlay = full_img.copy()
        overlay[algae_mask_pred > 0] = (0, 255, 0)
        overlay_path = os.path.join(water_data_dir, 'lake_algae_overlay.png')
        cv2.imwrite(overlay_path, overlay)
        print(f'Algae overlay is-- {overlay_path}')
    else:
        print("No regions found")

if __name__ == "__main__":
    latitude = 40.950065  
    longitude = -74.634119
    zoom = 2000

    print("Pulling latest images...")
    datapull_main(latitude, longitude, zoom)
    print("Image download complete. Running algae analysis.")
    analyze_lake()