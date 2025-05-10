import numpy as np
from skimage.measure import regionprops, label
from skimage.feature import graycomatrix, graycoprops

def extract_features(image, mask):

    if image is None or mask is None:
        raise ValueError("Image and mask cannot be None.")
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image must be a 3D array with 3 color channels.")
    if len(mask.shape) != 2:
        raise ValueError("Mask must be a 2D binary array.")

    labeled_mask = label(mask > 0)
    props = regionprops(labeled_mask)
    features = []

    for region in props:
        if region.area < 50:  # Skip small regions
            continue

        minr, minc, maxr, maxc = region.bbox
        patch = image[minr:maxr, minc:maxc]
        mask_patch = mask[minr:maxr, minc:maxc]

        color = patch[mask_patch > 0]
        if color.size == 0:
            continue
        color_mean = np.mean(color, axis=0)
        color_std = np.std(color, axis=0)

        try:
            gray = np.mean(patch, axis=2).astype(np.uint8)
            glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
        except Exception as e:
            print(f"Error computing GLCM: {e}")
            contrast = 0

        aspect = region.major_axis_length / (region.minor_axis_length + 1e-6)
        circularity = 4 * np.pi * region.area / (region.perimeter + 1e-6) ** 2

        features.append(np.concatenate([color_mean, color_std, [region.area, aspect, circularity, contrast]]))

    return np.vstack(features) if features else np.empty((0, 9))