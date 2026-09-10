import cv2
import numpy as np
import os
import json

def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image

def smart_upscale(image, target_width=3840, target_height=2160, model_path=None):
    h, w = image.shape[:2]
    scale_x = target_width / w
    scale_y = target_height / h
    scale = max(scale_x, scale_y)
    target_w = max(target_width, int(round(w * scale)))
    target_h = max(target_height, int(round(h * scale)))
    if model_path and os.path.exists(model_path):
        try:
            from cv2 import dnn_superres
            sr = dnn_superres.DnnSuperResImpl_create()
            sr.readModel(model_path)
            sr.setModel("edsr", 4)
            image = sr.upsample(image)
            if image.shape[1] != target_w or image.shape[0] != target_h:
                image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            return image
        except Exception:
            pass
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

def normalize_remote_sensing_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    image = cv2.bilateralFilter(image, d=7, sigmaColor=45, sigmaSpace=45)
    gaussian = cv2.GaussianBlur(image, (0, 0), 1.2)
    image = cv2.addWeighted(image, 1.45, gaussian, -0.45, 0)
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

def prepare_image(path, target_width=3840, target_height=2160, super_resolution_model=None):
    image = load_image(path)
    image = smart_upscale(image, target_width=target_width, target_height=target_height, model_path=super_resolution_model)
    image = normalize_remote_sensing_image(image)
    return image

def align_images(reference, moving):
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    moving_gray = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.GaussianBlur(reference_gray, (5, 5), 0)
    moving_gray = cv2.GaussianBlur(moving_gray, (5, 5), 0)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-7)
    try:
        _, warp_matrix = cv2.findTransformECC(reference_gray, moving_gray, warp_matrix, cv2.MOTION_AFFINE, criteria, None, 5)
        aligned = cv2.warpAffine(moving, warp_matrix, (reference.shape[1], reference.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
        return aligned
    except cv2.error:
        return cv2.resize(moving, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_LANCZOS4)

def radiometric_normalization(reference, moving):
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
    moving_lab = cv2.cvtColor(moving, cv2.COLOR_BGR2LAB)
    normalized = moving_lab.copy()
    for channel in range(3):
        ref_channel = reference_lab[:, :, channel].astype(np.float32)
        mov_channel = moving_lab[:, :, channel].astype(np.float32)
        ref_mean = np.mean(ref_channel)
        ref_std = np.std(ref_channel)
        mov_mean = np.mean(mov_channel)
        mov_std = np.std(mov_channel)
        if mov_std < 1e-6:
            mov_std = 1.0
        normalized_channel = ((mov_channel - mov_mean) / mov_std) * max(ref_std, 1.0) + ref_mean
        normalized[:, :, channel] = np.clip(normalized_channel, 0, 255).astype(np.uint8)
    return cv2.cvtColor(normalized, cv2.COLOR_LAB2BGR)

def calculate_multiscale_difference(before, after):
    before_lab = cv2.cvtColor(before, cv2.COLOR_BGR2LAB)
    after_lab = cv2.cvtColor(after, cv2.COLOR_BGR2LAB)
    scales = [(1.0, 3), (0.65, 7), (0.35, 15)]
    accumulated = np.zeros(before.shape[:2], dtype=np.float32)
    for weight, blur_size in scales:
        if blur_size == 3:
            before_scale = cv2.GaussianBlur(before_lab, (3, 3), 0)
            after_scale = cv2.GaussianBlur(after_lab, (3, 3), 0)
        else:
            before_scale = cv2.GaussianBlur(before_lab, (blur_size, blur_size), 0)
            after_scale = cv2.GaussianBlur(after_lab, (blur_size, blur_size), 0)
        difference = cv2.absdiff(before_scale, after_scale)
        l = difference[:, :, 0].astype(np.float32)
        a = difference[:, :, 1].astype(np.float32)
        b = difference[:, :, 2].astype(np.float32)
        score = 0.50 * l + 0.25 * a + 0.25 * b
        accumulated += weight * score
    accumulated = cv2.normalize(accumulated, None, 0, 255, cv2.NORM_MINMAX)
    return accumulated.astype(np.uint8)

def calculate_edge_difference(before, after):
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    before_edges = cv2.Canny(before_gray, 50, 150, apertureSize=3)
    after_edges = cv2.Canny(after_gray, 50, 150, apertureSize=3)
    before_edges = cv2.GaussianBlur(before_edges, (5, 5), 0)
    after_edges = cv2.GaussianBlur(after_edges, (5, 5), 0)
    return cv2.absdiff(before_edges, after_edges)

def generate_change_mask(pixel_difference, edge_difference, threshold=28, min_area=100):
    edge_difference = cv2.normalize(edge_difference, None, 0, 255, cv2.NORM_MINMAX)
    combined = 0.80 * pixel_difference.astype(np.float32) + 0.20 * edge_difference.astype(np.float32)
    combined = np.clip(combined, 0, 255).astype(np.uint8)
    percentile_threshold = np.percentile(combined, 92)
    adaptive_threshold = max(threshold, int(percentile_threshold))
    mask = (combined >= adaptive_threshold).astype(np.uint8) * 255
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    mask = cv2.dilate(mask, kernel_small, iterations=1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    regions = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        component = labels == i
        cleaned[component] = 255
        region_values = combined[component]
        score = float(np.mean(region_values))
        regions.append({
            "id": len(regions) + 1,
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "area_pixels": int(area),
            "centroid_x": float(centroids[i][0]),
            "centroid_y": float(centroids[i][1]),
            "change_score": round(score, 4)
        })
    regions.sort(key=lambda x: x["area_pixels"], reverse=True)
    for index, region in enumerate(regions, start=1):
        region["id"] = index
    return cleaned, combined, regions

def classify_change(before, after, region):
    x = region["x"]
    y = region["y"]
    w = region["width"]
    h = region["height"]
    before_roi = before[y:y + h, x:x + w]
    after_roi = after[y:y + h, x:x + w]
    if before_roi.size == 0 or after_roi.size == 0:
        return "localized visual change"
    before_hsv = cv2.cvtColor(before_roi, cv2.COLOR_BGR2HSV)
    after_hsv = cv2.cvtColor(after_roi, cv2.COLOR_BGR2HSV)
    green_before = ((before_hsv[:, :, 0] >= 25) & (before_hsv[:, :, 0] <= 95) & (before_hsv[:, :, 1] >= 35) & (before_hsv[:, :, 2] >= 30))
    green_after = ((after_hsv[:, :, 0] >= 25) & (after_hsv[:, :, 0] <= 95) & (after_hsv[:, :, 1] >= 35) & (after_hsv[:, :, 2] >= 30))
    vegetation_before = np.mean(green_before)
    vegetation_after = np.mean(green_after)
    gray_before = cv2.cvtColor(before_roi, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(after_roi, cv2.COLOR_BGR2GRAY)
    brightness_before = float(np.mean(gray_before))
    brightness_after = float(np.mean(gray_after))
    brightness_delta = brightness_after - brightness_before
    vegetation_delta = vegetation_after - vegetation_before
    if vegetation_delta > 0.15:
        change_type = "vegetation increase"
    elif vegetation_delta < -0.15:
        change_type = "vegetation decrease"
    elif brightness_delta > 25:
        change_type = "new bright surface or structure"
    elif brightness_delta < -25:
        change_type = "removed surface, darker structure, or shadow change"
    elif region["area_pixels"] > 10000:
        change_type = "large structural or land-cover change"
    elif region["area_pixels"] > 2500:
        change_type = "moderate structural or surface change"
    else:
        change_type = "localized surface change"
    region["change_type"] = change_type
    region["brightness_delta"] = round(brightness_delta, 4)
    region["vegetation_before"] = round(float(vegetation_before), 4)
    region["vegetation_after"] = round(float(vegetation_after), 4)
    region["vegetation_delta"] = round(float(vegetation_delta), 4)
    return change_type

def create_change_overlay(before, after, mask, regions):
    overlay = after.copy()
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    change_pixels = mask > 0
    overlay_float = overlay.astype(np.float32)
    heatmap_float = heatmap.astype(np.float32)
    overlay_float[change_pixels] = 0.35 * overlay_float[change_pixels] + 0.65 * heatmap_float[change_pixels]
    overlay = np.clip(overlay_float, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
    for region in regions:
        x = region["x"]
        y = region["y"]
        w = region["width"]
        h = region["height"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 2)
        label = f'{region["id"]} {region.get("change_type", "change")}'
        cv2.putText(overlay, label, (x, max(25, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    return overlay, heatmap

def analyze_two_images(before_path, after_path, target_width=3840, target_height=2160, super_resolution_model=None, threshold=28, min_region_area=100):
    before = prepare_image(before_path, target_width, target_height, super_resolution_model)
    after = prepare_image(after_path, target_width, target_height, super_resolution_model)
    after = cv2.resize(after, (before.shape[1], before.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    after = align_images(before, after)
    after = radiometric_normalization(before, after)
    after = normalize_remote_sensing_image(after)
    pixel_difference = calculate_multiscale_difference(before, after)
    edge_difference = calculate_edge_difference(before, after)
    mask, combined_score, regions = generate_change_mask(pixel_difference, edge_difference, threshold=threshold, min_area=min_region_area)
    for region in regions:
        classify_change(before, after, region)
    overlay, heatmap = create_change_overlay(before, after, mask, regions)
    total_pixels = before.shape[0] * before.shape[1]
    changed_pixels = int(np.count_nonzero(mask))
    change_percentage = (changed_pixels / total_pixels) * 100.0
    result = {
        "before": before,
        "after": after,
        "difference": pixel_difference,
        "combined_score": combined_score,
        "mask": mask,
        "heatmap": heatmap,
        "overlay": overlay,
        "regions": regions,
        "image_width": int(before.shape[1]),
        "image_height": int(before.shape[0]),
        "total_pixels": int(total_pixels),
        "changed_pixels": int(changed_pixels),
        "change_percentage": round(change_percentage, 4)
    }
    return result

def compare_satellite_images(before_path, after_path):
    result = analyze_two_images(before_path=before_path, after_path=after_path, target_width=1920, target_height=1080, super_resolution_model=None, threshold=28, min_region_area=100)
    return {
        "changed_pixels": result["changed_pixels"],
        "change_percentage": result["change_percentage"],
        "number_of_change_regions": len(result["regions"]),
        "changes": result["regions"],
        "before_image": result["before"],
        "after_image": result["after"],
        "change_mask": result["mask"],
        "change_heatmap": result["heatmap"],
        "change_overlay": result["overlay"]
    }
