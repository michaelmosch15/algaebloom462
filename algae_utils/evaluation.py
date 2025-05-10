from sklearn.metrics import f1_score, jaccard_score

def compute_iou(mask_pred, mask_true):
    mask_pred = (mask_pred > 0).astype(int).ravel()
    mask_true = (mask_true > 0).astype(int).ravel()
    return jaccard_score(mask_true, mask_pred)

def compute_f1(mask_pred, mask_true):
    mask_pred = (mask_pred > 0).astype(int).ravel()
    mask_true = (mask_true > 0).astype(int).ravel()
    return f1_score(mask_true, mask_pred)