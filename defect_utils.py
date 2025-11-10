import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
import random

def generate_defect_boxes(image_path, reference_folder="reference_ok_images"):
    """Detect scratches, dents, uneven edges, and shape irregularities only."""
    
    # --- Load and preprocess images ---
    img = cv2.imread(image_path)
    ok_images = [os.path.join(reference_folder, f) for f in os.listdir(reference_folder) 
                 if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    ref_path = random.choice(ok_images)
    ref = cv2.imread(ref_path)
    ref = cv2.resize(ref, (img.shape[1], img.shape[0]))

    # --- Convert both to grayscale ---
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    # --- Normalize lighting and contrast ---
    gray_img = cv2.equalizeHist(gray_img)
    gray_ref = cv2.equalizeHist(gray_ref)

    # --- Apply Gaussian blur to remove minor illumination differences ---
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    gray_ref = cv2.GaussianBlur(gray_ref, (5, 5), 0)

    # --- SSIM comparison (focus on structure not brightness) ---
    score, diff = ssim(gray_ref, gray_img, full=True)
    diff = (1 - diff) * 255
    diff = diff.astype("uint8")

    # --- Threshold and morphological cleanup ---
    _, thresh = cv2.threshold(diff, 60, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph = cv2.GaussianBlur(morph, (3, 3), 0)

    # --- Edge detection for scratches / irregularities ---
    edges = cv2.Canny(gray_img, 80, 180)
    edge_mask = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # --- Combine edge and morphological results ---
    combined = cv2.bitwise_and(morph, morph, mask=edge_mask)

    # --- Find and draw bounding boxes ---
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if 80 < area < (0.03 * img.shape[0] * img.shape[1]):  # filters small noise
            x, y, bw, bh = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    return img, score
