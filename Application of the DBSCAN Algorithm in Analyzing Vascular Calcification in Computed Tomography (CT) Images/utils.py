import os
import numpy as np
import cv2
import pydicom
from PIL import Image

def load_image_file(file_path):
    """
    Đọc ảnh DICOM hoặc ảnh thường.
    Trả về:
      - hu_matrix: mảng kích thước thực (Hounsfield Unit cho DICOM, giá trị pixel cho ảnh thường)
      - img_display: mảng chuẩn hóa 0-255 dùng để hiển thị giao diện.
    """
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".dcm":
        dicom = pydicom.dcmread(file_path)
        img_raw = dicom.pixel_array.astype(np.float32)
        intercept = float(getattr(dicom, 'RescaleIntercept', 0.0))
        slope = float(getattr(dicom, 'RescaleSlope', 1.0))
        
        # Giá trị không gian HU vật lý thật để phân tích y khoa (Bản sửa lỗi)
        hu_matrix = img_raw * slope + intercept
        
        # Mảng 8-bit chỉ dùng cho hiển thị hình ảnh
        img_norm = ((hu_matrix - np.min(hu_matrix)) / (np.max(hu_matrix) - np.min(hu_matrix)) * 255.0)
        img_display = np.clip(img_norm, 0, 255).astype(np.uint8)
    else:
        img_pil = Image.open(file_path).convert('L')
        img_display = np.array(img_pil).astype(np.uint8)
        # Giả định matrix HU là giá trị màu cho ảnh không phải DICOM
        hu_matrix = img_display.astype(np.float32)
        
    return hu_matrix, img_display

def skull_strip_and_crop(img_gray, padding=10):
    """
    Tách bỏ nhiễu ngoại biên và vùng sọ não (nếu có).
    """
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel2, iterations=1)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        h, w = img_gray.shape
        mask = np.ones_like(img_gray)*255
        return mask, (0, h, 0, w), (0,0), mask
        
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(img_gray)
    cv2.drawContours(mask, [largest], -1, 255, thickness=-1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2, iterations=1)
    x, y, w, h = cv2.boundingRect(largest)
    
    y0 = max(0, y - padding)
    y1 = min(img_gray.shape[0], y + h + padding)
    x0 = max(0, x - padding)
    x1 = min(img_gray.shape[1], x + w + padding)
    
    masked = img_gray.copy()
    masked[mask == 0] = 0
    return masked, (y0, y1, x0, x1), (y0, x0), mask

def preprocess_image_pipeline(display_image):
    """
    Pipline tiền xử lý ảnh, tạo ảnh nhị phân và ảnh mask cuối cùng.
    """
    img = display_image.copy()
    masked, bbox, offset, skull_mask = skull_strip_and_crop(img, padding=8)
    y0, y1, x0, x1 = bbox
    
    cropped_masked = masked[y0:y1, x0:x1]
    if cropped_masked.size == 0:
        cropped_masked = masked.copy()
        y0, x0 = 0, 0
        offset = (0, 0)
        
    blur2 = cv2.GaussianBlur(cropped_masked, (3,3), 0)
    if np.max(blur2) == 0:
        _, bin_img = cv2.threshold(cropped_masked, 127, 255, cv2.THRESH_BINARY)
    else:
        _, bin_img = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel_small, iterations=2)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    
    global_mask = np.zeros_like(img)
    crop_mask = (cropped_masked > 0).astype(np.uint8) * 255
    global_mask[y0:y1, x0:x1] = crop_mask
    
    full_binary = np.zeros_like(img, dtype=np.uint8)
    full_binary[y0:y1, x0:x1] = bin_img
    full_binary[skull_mask == 0] = 0  # Re-apply skull mask if out of region
    
    num_labels, labels_im = cv2.connectedComponents(full_binary)
    final_bin = np.zeros_like(full_binary)
    min_comp_size = max(8, int(0.00005 * img.size))
    
    for lab in range(1, num_labels):
        comp_mask = (labels_im == lab)
        comp_size = np.sum(comp_mask)
        if comp_size >= min_comp_size:
            final_bin[comp_mask] = 255
            
    image_binary = final_bin.copy()
    
    # image_data is skull-stripped for display logic
    image_data = img.copy()
    image_data[skull_mask == 0] = 0
    
    overlay = image_data.copy()
    overlay[y0:y1, x0:x1][bin_img > 0] = 255
    
    return image_binary, image_data, overlay, offset, bbox

def calculate_circularity(coords):
    if len(coords) < 3:
        return 0
    try:
        min_x, max_x = np.min(coords[:, 0]), np.max(coords[:, 0])
        min_y, max_y = np.min(coords[:, 1]), np.max(coords[:, 1])
        mask = np.zeros((max_x - min_x + 1, max_y - min_y + 1), dtype=np.uint8)
        for x, y in coords:
            mask[x - min_x, y - min_y] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = contours[0]
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                return min(circularity, 1.0)
        return 0
    except Exception:
        return 0
