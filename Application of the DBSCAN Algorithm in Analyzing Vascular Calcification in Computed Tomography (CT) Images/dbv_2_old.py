# File: DBV2.py
# Phiên bản DBSCAN - DBV2: tiền xử lý loại bỏ metadata ngoài lề
# Yêu cầu: Python 3.x, OpenCV (cv2), numpy, pydicom, scikit-learn, PyQt6, Pillow

import sys
import numpy as np
import cv2
import pydicom
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from PyQt6.QtWidgets import (
    QApplication, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QHBoxLayout, QSlider, QSpinBox, QGridLayout,
    QTextEdit, QFrame, QMessageBox
)
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import Qt
import os
from PIL import Image


class DBSCANApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Phân tích vôi hóa mạch máu - DBV2")
        self.setGeometry(100, 100, 1600, 800)

        # Biến lưu trữ
        self.image_path = None
        self.image_data = None       # grayscale image (0-255), masked skull-stripped version
        self.image_binary = None     # nhị phân, same size as original
        self.original_image = None   # gốc đã được chuẩn hoá (0-255)
        self.result_img = None
        self.result_img_color = None
        self.clustering_labels = None
        self.coords = None
        self.crop_offset = (0, 0)    # offset nếu chúng ta crop

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()

        left_panel = QVBoxLayout()

        self.label_original = QLabel("Ảnh gốc")
        self.label_preprocessed = QLabel("Ảnh tiền xử lý")
        self.label_result = QLabel("Kết quả phân tích")

        for label in [self.label_original, self.label_preprocessed, self.label_result]:
            label.setFixedSize(350, 350)
            label.setStyleSheet("border: 2px solid #333; background-color: #f0f0f0;")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setFont(QFont("Arial", 10))

        image_layout_top = QHBoxLayout()
        image_layout_top.addWidget(self.label_original)
        image_layout_top.addWidget(self.label_preprocessed)

        image_layout_bottom = QHBoxLayout()
        image_layout_bottom.addWidget(self.label_result)

        left_panel.addLayout(image_layout_top)
        left_panel.addLayout(image_layout_bottom)

        self.btn_open = QPushButton("Chọn ảnh")
        self.btn_process = QPushButton("Tiền xử lý")
        self.btn_dbscan = QPushButton("Phân tích DBSCAN")
        self.btn_save_result = QPushButton("Lưu kết quả")
        self.btn_save_color = QPushButton("Lưu ảnh màu")

        for btn in [self.btn_open, self.btn_process, self.btn_dbscan, self.btn_save_result, self.btn_save_color]:
            btn.setFixedHeight(40)
            btn.setFont(QFont("Arial", 10))

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_open)
        button_layout.addWidget(self.btn_process)
        button_layout.addWidget(self.btn_dbscan)
        button_layout.addWidget(self.btn_save_result)
        button_layout.addWidget(self.btn_save_color)

        left_panel.addLayout(button_layout)

        param_frame = QFrame()
        param_frame.setFrameStyle(QFrame.Shape.Box)
        param_frame.setStyleSheet("QFrame { border: 1px solid #ccc; padding: 10px; }")
        param_layout = QVBoxLayout(param_frame)

        param_title = QLabel("Tham số DBSCAN")
        param_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        param_layout.addWidget(param_title)

        eps_layout = QHBoxLayout()
        eps_layout.addWidget(QLabel("Epsilon (eps):"))
        self.slider_eps = QSlider(Qt.Orientation.Horizontal)
        self.slider_eps.setMinimum(1)
        self.slider_eps.setMaximum(20)
        self.slider_eps.setValue(5)
        self.label_eps_value = QLabel("5")
        self.slider_eps.valueChanged.connect(lambda v: self.label_eps_value.setText(str(v)))
        eps_layout.addWidget(self.slider_eps)
        eps_layout.addWidget(self.label_eps_value)
        param_layout.addLayout(eps_layout)

        minpts_layout = QHBoxLayout()
        minpts_layout.addWidget(QLabel("Min Points:"))
        self.spin_minpts = QSpinBox()
        self.spin_minpts.setMinimum(1)
        self.spin_minpts.setMaximum(50)
        self.spin_minpts.setValue(5)
        minpts_layout.addWidget(self.spin_minpts)
        param_layout.addLayout(minpts_layout)

        left_panel.addWidget(param_frame)

        right_panel = QVBoxLayout()

        result_title = QLabel("Kết quả phân tích chi tiết")
        result_title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        right_panel.addWidget(result_title)

        self.text_result = QTextEdit()
        self.text_result.setFixedSize(400, 600)
        self.text_result.setFont(QFont("Courier", 10))
        self.text_result.setReadOnly(True)
        right_panel.addWidget(self.text_result)

        main_layout.addLayout(left_panel)
        main_layout.addLayout(right_panel)

        self.setLayout(main_layout)

        # Kết nối
        self.btn_open.clicked.connect(self.load_image)
        self.btn_process.clicked.connect(self.preprocess_image)
        self.btn_dbscan.clicked.connect(self.run_dbscan)
        self.btn_save_result.clicked.connect(self.save_result_image)
        self.btn_save_color.clicked.connect(self.save_color_image)

        self.btn_process.setEnabled(False)
        self.btn_dbscan.setEnabled(False)
        self.btn_save_result.setEnabled(False)
        self.btn_save_color.setEnabled(False)

    def show_message(self, title, message, msg_type="info"):
        msg_box = QMessageBox()
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        if msg_type == "error":
            msg_box.setIcon(QMessageBox.Icon.Critical)
        elif msg_type == "warning":
            msg_box.setIcon(QMessageBox.Icon.Warning)
        else:
            msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.exec()

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh", "",
            "Tất cả ảnh (*.dcm *.png *.jpg *.jpeg *.bmp);;DICOM (*.dcm);;Ảnh thường (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_path:
            return
        try:
            ext = os.path.splitext(file_path)[-1].lower()
            if ext == ".dcm":
                dicom = pydicom.dcmread(file_path)
                img = dicom.pixel_array.astype(np.float32)
                intercept = float(getattr(dicom, 'RescaleIntercept', 0.0))
                slope = float(getattr(dicom, 'RescaleSlope', 1.0))
                img = img * slope + intercept
                img_norm = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255.0)
                img = np.clip(img_norm, 0, 255).astype(np.uint8)
            else:
                img_pil = Image.open(file_path).convert('L')
                img = np.array(img_pil).astype(np.uint8)

            self.original_image = img.copy()
            self.image_data = img.copy()  # mặc định image_data = original, sẽ mask trong tiền xử lý
            self.image_binary = None
            self.result_img = None
            self.result_img_color = None
            self.clustering_labels = None
            self.coords = None
            self.crop_offset = (0, 0)

            self.display_image(self.label_original, img)
            self.label_preprocessed.clear()
            self.label_preprocessed.setText("Ảnh tiền xử lý")
            self.label_result.clear()
            self.label_result.setText("Kết quả phân tích")

            self.btn_process.setEnabled(True)
            self.btn_dbscan.setEnabled(False)
            self.btn_save_result.setEnabled(False)
            self.btn_save_color.setEnabled(False)

            self.text_result.clear()
            self.text_result.append(f"Đã tải ảnh: {os.path.basename(file_path)}")
            self.text_result.append(f"Kích thước: {img.shape[1]} x {img.shape[0]} pixels")
            self.text_result.append(f"Giá trị pixel: {np.min(img)} - {np.max(img)}")

        except Exception as e:
            self.show_message("Lỗi", f"Không thể đọc ảnh!\nLỗi: {str(e)}", "error")

    def skull_strip_and_crop(self, img_gray, padding=10):
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
            return mask, (0, h, 0, w), (0,0)
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
        return masked, (y0, y1, x0, x1), (y0, x0)

    def preprocess_image(self):
        if self.image_data is None:
            self.show_message("Cảnh báo", "Chưa tải ảnh!", "warning")
            return
        try:
            img = self.image_data.copy()
            masked, bbox, offset = self.skull_strip_and_crop(img, padding=8)
            self.crop_offset = offset
            y0, y1, x0, x1 = bbox
            cropped_masked = masked[y0:y1, x0:x1]
            if cropped_masked.size == 0:
                cropped_masked = masked.copy()
                y0, x0 = 0, 0
                self.crop_offset = (0, 0)
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
            full_binary[global_mask == 0] = 0
            num_labels, labels_im = cv2.connectedComponents(full_binary)
            final_bin = np.zeros_like(full_binary)
            min_comp_size = max(8, int(0.00005 * img.size))
            for lab in range(1, num_labels):
                comp_mask = (labels_im == lab)
                comp_size = np.sum(comp_mask)
                if comp_size >= min_comp_size:
                    final_bin[comp_mask] = 255
            self.image_binary = final_bin.copy()
            preprocessed_vis = img.copy()
            preprocessed_vis[global_mask == 0] = 0
            overlay = preprocessed_vis.copy()
            overlay[y0:y1, x0:x1][bin_img > 0] = 255
            # Update image_data to the masked version for all further operations
            # we keep original_image intact but image_data will be the skull-stripped image
            self.image_data = preprocessed_vis
            self.display_image(self.label_preprocessed, overlay)
            self.btn_dbscan.setEnabled(True)
            white_pixels = np.sum(self.image_binary == 255)
            total_pixels = self.image_binary.shape[0] * self.image_binary.shape[1]
            self.text_result.append("\nTiền xử lý hoàn thành (DBV2):")
            self.text_result.append(f"Pixel trắng (ứng viên): {white_pixels:,}")
            self.text_result.append(f"Tỷ lệ: {white_pixels/total_pixels*100:.4f}%")
            self.text_result.append(f"Crop offset: {self.crop_offset}")
            self.text_result.append(f"Bounding box: ({x0},{y0}) - ({x1},{y1})")

        except Exception as e:
            self.show_message("Lỗi", f"Lỗi khi tiền xử lý!\nLỗi: {str(e)}", "error")

    def run_dbscan(self):
        if self.image_binary is None:
            self.show_message("Cảnh báo", "Chưa tiền xử lý ảnh!", "warning")
            return
        try:
            eps = self.slider_eps.value()
            min_pts = self.spin_minpts.value()
            coords = np.column_stack(np.where(self.image_binary > 0))
            if coords.shape[0] == 0:
                self.show_message("Cảnh báo", "Không tìm thấy pixel trắng nào để phân tích!", "warning")
                return
            self.coords = coords
            self.text_result.append(f"\n🔍 Bắt đầu phân tích DBSCAN...")
            self.text_result.append(f"📍 Số điểm cần phân tích: {coords.shape[0]:,}")
            self.text_result.append(f"⚙️ Tham số: eps={eps}, min_samples={min_pts}")
            clustering = DBSCAN(eps=eps, min_samples=min_pts).fit(coords)
            self.clustering_labels = clustering.labels_
            self.result_img = self.create_result_image()
            self.display_image(self.label_result, self.result_img)
            self.analyze_clusters()
            self.btn_save_result.setEnabled(True)
            self.btn_save_color.setEnabled(True)
        except Exception as e:
            self.show_message("Lỗi", f"Lỗi khi phân tích DBSCAN!\nLỗi: {str(e)}", "error")

    def create_result_image(self):
        """Tạo ảnh kết quả với các cụm được đánh dấu (chỉ hiển thị trên vùng sọ đã mask)"""
        if self.clustering_labels is None or self.coords is None:
            return None

        # Dùng ảnh đã mask (image_data) làm nền
        height, width = self.image_data.shape
        result_img = np.zeros((height, width, 3), dtype=np.uint8)
        result_img[:, :, 0] = self.image_data
        result_img[:, :, 1] = self.image_data
        result_img[:, :, 2] = self.image_data

        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (255, 128, 0),
            (128, 0, 255), (255, 192, 203), (128, 128, 128)
        ]

        unique_labels = sorted([label for label in set(self.clustering_labels) if label != -1])

        for i, label in enumerate(unique_labels):
            cluster_coords = self.coords[self.clustering_labels == label]
            color = colors[i % len(colors)]

            for x, y in cluster_coords:
                if 0 <= x < height and 0 <= y < width:
                    result_img[x, y] = color

            min_x, max_x = np.min(cluster_coords[:, 0]), np.max(cluster_coords[:, 0])
            min_y, max_y = np.min(cluster_coords[:, 1]), np.max(cluster_coords[:, 1])

            cv2.rectangle(result_img, (min_y-2, min_x-2), (max_y+2, max_x+2), color, 2)
            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2
            cv2.circle(result_img, (center_y, center_x), 15, (255, 255, 255), -1)
            cv2.circle(result_img, (center_y, center_x), 15, color, 2)
            cv2.putText(result_img, str(i+1), (center_y-8, center_x+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        noise_coords = self.coords[self.clustering_labels == -1]
        for x, y in noise_coords:
            if 0 <= x < height and 0 <= y < width:
                result_img[x, y] = (64, 64, 64)

        self.result_img_color = result_img.copy()
        result_gray = cv2.cvtColor(result_img, cv2.COLOR_RGB2GRAY)
        return result_gray

    def analyze_clusters(self):
        """Phân tích chi tiết các cụm"""
        if self.clustering_labels is None or self.coords is None:
            return

        unique_labels = set(self.clustering_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(self.clustering_labels).count(-1)

        self.text_result.append(f"\nKẾT QUẢ PHÂN TÍCH:")
        self.text_result.append(f"Số cụm tìm được: {n_clusters}")
        self.text_result.append(f"Điểm nhiễu: {n_noise}")
        self.text_result.append(f"Tổng điểm: {len(self.clustering_labels)}")

        if n_clusters == 0:
            self.text_result.append("⚠️ Không tìm thấy cụm nào. Hãy thử điều chỉnh tham số.")
            return

        self.text_result.append(f"\n{'='*50}")
        self.text_result.append("CHI TIẾT CÁC CỤM:")

        height, width = self.image_data.shape  # Dùng ảnh đã mask sạch

        for label in sorted(unique_labels):
            if label == -1:
                continue

            cluster_coords = self.coords[self.clustering_labels == label]
            cluster_size = len(cluster_coords)

            hu_values = []
            for x, y in cluster_coords:
                if 0 <= x < height and 0 <= y < width:
                    hu_values.append(int(self.image_data[x, y]))

            if hu_values:
                mean_hu = np.mean(hu_values)
                std_hu = np.std(hu_values)
                min_hu = np.min(hu_values)
                max_hu = np.max(hu_values)
            else:
                mean_hu = std_hu = min_hu = max_hu = 0

            circularity = self.calculate_circularity(cluster_coords)

            min_x, max_x = np.min(cluster_coords[:, 0]), np.max(cluster_coords[:, 0])
            min_y, max_y = np.min(cluster_coords[:, 1]), np.max(cluster_coords[:, 1])
            bbox_width = max_y - min_y + 1
            bbox_height = max_x - min_x + 1

            self.text_result.append(f"\nCỤM #{label + 1}:")
            self.text_result.append(f"  Kích thước: {cluster_size} pixels")
            self.text_result.append(f"  HU trung bình: {mean_hu:.1f} ± {std_hu:.1f}")
            self.text_result.append(f"  HU min-max: {min_hu} - {max_hu}")
            self.text_result.append(f"  Circularity: {circularity:.3f}")
            self.text_result.append(f"  Bounding box: {bbox_width}x{bbox_height}")
            self.text_result.append(f"  Vị trí: ({min_x},{min_y}) - ({max_x},{max_y})")

            if mean_hu > 130:
                confidence = "CAO"
            elif mean_hu > 100:
                confidence = "TRUNG BÌNH"
            else:
                confidence = "THẤP"

            self.text_result.append(f"  Khả năng vôi hóa: {confidence}")

    def calculate_circularity(self, coords):
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

    def save_result_image(self):
        if self.result_img is None:
            self.show_message("Cảnh báo", "Chưa có kết quả để lưu!", "warning")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Lưu ảnh kết quả", "dbscan_result.png",
            "Images (*.png *.jpg *.bmp)"
        )
        if file_path:
            try:
                cv2.imwrite(file_path, self.result_img)
                self.show_message("Thành công", f"Đã lưu ảnh kết quả tại:\n{file_path}")
                report_path = file_path.replace('.png', '_report.txt').replace('.jpg', '_report.txt').replace('.bmp', '_report.txt')
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(self.text_result.toPlainText())
                self.text_result.append(f"\nĐã lưu báo cáo tại: {report_path}")
            except Exception as e:
                self.show_message("Lỗi", f"Không thể lưu file!\nLỗi: {str(e)}", "error")

    def save_color_image(self):
        """Lưu ảnh màu với các cụm được đánh dấu (nền đã loại bỏ metadata)"""
        if self.result_img_color is None:
            self.show_message("Cảnh báo", "Chưa có kết quả màu để lưu!", "warning")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Lưu ảnh màu", "dbscan_color_result.png",
            "Images (*.png *.jpg *.bmp)"
        )
        if file_path:
            try:
                img_bgr = cv2.cvtColor(self.result_img_color, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, img_bgr)
                self.show_message("Thành công", f"Đã lưu ảnh màu tại:\n{file_path}")
                legend_path = file_path.replace('.png', '_legend.txt').replace('.jpg', '_legend.txt').replace('.bmp', '_legend.txt')
                self.create_color_legend(legend_path)
                self.text_result.append(f"\n Đã lưu ảnh màu tại: {file_path}")
                self.text_result.append(f"Đã lưu chú thích màu tại: {legend_path}")
            except Exception as e:
                self.show_message("Lỗi", f"Không thể lưu file màu!\nLỗi: {str(e)}", "error")

    def create_color_legend(self, legend_path):
        colors = [
            "Đỏ", "Xanh lá", "Xanh dương", "Vàng", "Tím",
            "Cyan", "Cam", "Tím đậm", "Hồng", "Xám"
        ]
        try:
            with open(legend_path, 'w', encoding='utf-8') as f:
                f.write("CHƯƠNG TRÌNH PHÂN TÍCH VÔI HÓA MẠCH MÁU - DBV2\n")
                f.write("="*50 + "\n\n")
                f.write("CHÚ THÍCH MÀU SẮC CÁC CỤM:\n")
                f.write("-" * 30 + "\n")
                unique_labels = sorted([label for label in set(self.clustering_labels) if label != -1])
                for i, label in enumerate(unique_labels):
                    cluster_coords = self.coords[self.clustering_labels == label]
                    cluster_size = len(cluster_coords)
                    color_name = colors[i % len(colors)]
                    f.write(f"Cụm #{i+1}: {color_name} ({cluster_size} pixels)\n")
                f.write(f"\nĐiểm nhiễu: Xám đậm\n")
                f.write(f"Khung viền: Bao quanh mỗi cụm\n")
                f.write(f"Số thứ tự: Hiển thị ở giữa mỗi cụm\n")
        except Exception as e:
            print(f"Không thể tạo file chú thích: {e}")

    def display_image(self, target_label, img):
        if img is None:
            return
        height, width = img.shape
        bytes_per_line = width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img).scaled(
            350, 350, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        target_label.setPixmap(pixmap)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = DBSCANApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
