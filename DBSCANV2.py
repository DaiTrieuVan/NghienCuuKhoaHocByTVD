import sys
import numpy as np
import cv2
import pydicom
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from PyQt6.QtWidgets import (
    QApplication, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QHBoxLayout, QSlider, QSpinBox, QGridLayout,
    QTextEdit, QScrollArea, QFrame, QMessageBox, QCheckBox, QGroupBox
)
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import Qt
import os
from PIL import Image


class DBSCANApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Phân tích vôi hóa mạch máu - DBSCAN với ROI Detection")
        self.setGeometry(100, 100, 1600, 900)

        # Khởi tạo biến
        self.image_path = None
        self.image_data = None
        self.image_binary = None
        self.original_image = None
        self.roi_mask = None
        self.result_img = None
        self.result_img_color = None
        self.clustering_labels = None
        self.coords = None

        self.init_ui()

    def init_ui(self):
        # Layout chính
        main_layout = QHBoxLayout()

        # Phần bên trái - Hiển thị ảnh
        left_panel = QVBoxLayout()
        
        # Nhãn ảnh - thêm ảnh ROI
        self.label_original = QLabel("Ảnh gốc")
        self.label_roi = QLabel("Vùng quan tâm (ROI)")
        self.label_preprocessed = QLabel("Ảnh tiền xử lý")
        self.label_result = QLabel("Kết quả phân tích")
        
        for label in [self.label_original, self.label_roi, self.label_preprocessed, self.label_result]:
            label.setFixedSize(350, 350)
            label.setStyleSheet("border: 2px solid #333; background-color: #f0f0f0;")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setFont(QFont("Arial", 10))

        # Layout ảnh 2x2
        image_layout_top = QHBoxLayout()
        image_layout_top.addWidget(self.label_original)
        image_layout_top.addWidget(self.label_roi)
        
        image_layout_bottom = QHBoxLayout()
        image_layout_bottom.addWidget(self.label_preprocessed)
        image_layout_bottom.addWidget(self.label_result)
        
        left_panel.addLayout(image_layout_top)
        left_panel.addLayout(image_layout_bottom)

        # Nút điều khiển
        self.btn_open = QPushButton("📁 Chọn ảnh")
        self.btn_detect_roi = QPushButton("🎯 Phát hiện ROI")
        self.btn_process = QPushButton("🔧 Tiền xử lý")
        self.btn_dbscan = QPushButton("🔍 Phân tích DBSCAN")
        self.btn_save_result = QPushButton("💾 Lưu kết quả")
        self.btn_save_color = QPushButton("🎨 Lưu ảnh màu")

        for btn in [self.btn_open, self.btn_detect_roi, self.btn_process, self.btn_dbscan, self.btn_save_result, self.btn_save_color]:
            btn.setFixedHeight(40)
            btn.setFont(QFont("Arial", 10))

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_open)
        button_layout.addWidget(self.btn_detect_roi)
        button_layout.addWidget(self.btn_process)
        button_layout.addWidget(self.btn_dbscan)
        button_layout.addWidget(self.btn_save_result)
        button_layout.addWidget(self.btn_save_color)
        
        left_panel.addLayout(button_layout)

        # Tham số ROI Detection
        roi_frame = QFrame()
        roi_frame.setFrameStyle(QFrame.Shape.Box)
        roi_frame.setStyleSheet("QFrame { border: 1px solid #ccc; padding: 10px; }")
        roi_layout = QVBoxLayout(roi_frame)

        roi_title = QLabel("Tham số ROI Detection")
        roi_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        roi_layout.addWidget(roi_title)

        # Checkbox tự động phát hiện ROI
        self.check_auto_roi = QCheckBox("Tự động phát hiện vùng ROI")
        self.check_auto_roi.setChecked(True)
        roi_layout.addWidget(self.check_auto_roi)

        # Slider ROI threshold
        roi_thresh_layout = QHBoxLayout()
        roi_thresh_layout.addWidget(QLabel("ROI Threshold:"))
        self.slider_roi_thresh = QSlider(Qt.Orientation.Horizontal)
        self.slider_roi_thresh.setMinimum(50)
        self.slider_roi_thresh.setMaximum(200)
        self.slider_roi_thresh.setValue(100)
        self.label_roi_thresh_value = QLabel("100")
        self.slider_roi_thresh.valueChanged.connect(lambda v: self.label_roi_thresh_value.setText(str(v)))
        roi_thresh_layout.addWidget(self.slider_roi_thresh)
        roi_thresh_layout.addWidget(self.label_roi_thresh_value)
        roi_layout.addLayout(roi_thresh_layout)

        # Slider minimum area
        min_area_layout = QHBoxLayout()
        min_area_layout.addWidget(QLabel("Min ROI Area:"))
        self.slider_min_area = QSlider(Qt.Orientation.Horizontal)
        self.slider_min_area.setMinimum(1000)
        self.slider_min_area.setMaximum(50000)
        self.slider_min_area.setValue(10000)
        self.label_min_area_value = QLabel("10000")
        self.slider_min_area.valueChanged.connect(lambda v: self.label_min_area_value.setText(str(v)))
        min_area_layout.addWidget(self.slider_min_area)
        min_area_layout.addWidget(self.label_min_area_value)
        roi_layout.addLayout(min_area_layout)

        left_panel.addWidget(roi_frame)

        # Tham số DBSCAN
        param_frame = QFrame()
        param_frame.setFrameStyle(QFrame.Shape.Box)
        param_frame.setStyleSheet("QFrame { border: 1px solid #ccc; padding: 10px; }")
        param_layout = QVBoxLayout(param_frame)

        param_title = QLabel("Tham số DBSCAN")
        param_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        param_layout.addWidget(param_title)

        # Epsilon slider
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

        # MinPts spinbox
        minpts_layout = QHBoxLayout()
        minpts_layout.addWidget(QLabel("Min Points:"))
        self.spin_minpts = QSpinBox()
        self.spin_minpts.setMinimum(1)
        self.spin_minpts.setMaximum(50)
        self.spin_minpts.setValue(5)
        minpts_layout.addWidget(self.spin_minpts)
        param_layout.addLayout(minpts_layout)

        left_panel.addWidget(param_frame)

        # Phần bên phải - Kết quả phân tích
        right_panel = QVBoxLayout()
        
        result_title = QLabel("Kết quả phân tích chi tiết")
        result_title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        right_panel.addWidget(result_title)

        self.text_result = QTextEdit()
        self.text_result.setFixedSize(400, 700)
        self.text_result.setFont(QFont("Courier", 10))
        self.text_result.setReadOnly(True)
        right_panel.addWidget(self.text_result)

        # Thêm vào layout chính
        main_layout.addLayout(left_panel)
        main_layout.addLayout(right_panel)
        
        self.setLayout(main_layout)

        # Kết nối sự kiện
        self.btn_open.clicked.connect(self.load_image)
        self.btn_detect_roi.clicked.connect(self.detect_roi)
        self.btn_process.clicked.connect(self.preprocess_image)
        self.btn_dbscan.clicked.connect(self.run_dbscan)
        self.btn_save_result.clicked.connect(self.save_result_image)
        self.btn_save_color.clicked.connect(self.save_color_image)

        # Cập nhật trạng thái ban đầu
        self.btn_detect_roi.setEnabled(False)
        self.btn_process.setEnabled(False)
        self.btn_dbscan.setEnabled(False)
        self.btn_save_result.setEnabled(False)
        self.btn_save_color.setEnabled(False)

    def show_message(self, title, message, msg_type="info"):
        """Hiển thị thông báo"""
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
        """Tải ảnh từ file"""
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
                # Chuẩn hóa về 0-255
                img = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
            else:
                img_pil = Image.open(file_path).convert('L')
                img = np.array(img_pil).astype(np.uint8)
            
            self.original_image = img.copy()
            self.image_data = img.copy()
            self.image_binary = None
            self.roi_mask = None
            self.result_img = None
            self.result_img_color = None
            self.clustering_labels = None
            self.coords = None
            
            # Hiển thị ảnh
            self.display_image(self.label_original, img)
            self.label_roi.clear()
            self.label_roi.setText("Vùng quan tâm (ROI)")
            self.label_preprocessed.clear()
            self.label_preprocessed.setText("Ảnh tiền xử lý")
            self.label_result.clear()
            self.label_result.setText("Kết quả phân tích")
            
            # Cập nhật trạng thái nút
            self.btn_detect_roi.setEnabled(True)
            self.btn_process.setEnabled(False)
            self.btn_dbscan.setEnabled(False)
            self.btn_save_result.setEnabled(False)
            self.btn_save_color.setEnabled(False)
            
            # Xóa kết quả cũ
            self.text_result.clear()
            self.text_result.append(f"✅ Đã tải ảnh: {os.path.basename(file_path)}")
            self.text_result.append(f"📏 Kích thước: {img.shape[1]} x {img.shape[0]} pixels")
            self.text_result.append(f"🎯 Giá trị pixel: {np.min(img)} - {np.max(img)}")
            
            # Tự động phát hiện ROI nếu được bật
            if self.check_auto_roi.isChecked():
                self.detect_roi()
                
        except Exception as e:
            self.show_message("Lỗi", f"Không thể đọc ảnh!\nLỗi: {str(e)}", "error")

    def detect_roi(self):
        """Phát hiện vùng quan tâm (ROI) trong ảnh CT"""
        if self.image_data is None:
            self.show_message("Cảnh báo", "Chưa tải ảnh!", "warning")
            return

        try:
            roi_threshold = self.slider_roi_thresh.value()
            min_area = self.slider_min_area.value()
            
            self.text_result.append(f"\n🎯 Bắt đầu phát hiện ROI...")
            self.text_result.append(f"⚙️ Threshold: {roi_threshold}, Min area: {min_area}")
            
            # Tạo mask ban đầu dựa trên threshold
            _, roi_binary = cv2.threshold(self.image_data, roi_threshold, 255, cv2.THRESH_BINARY)
            
            # Loại bỏ các vùng nhỏ (nhiễu, text)
            kernel = np.ones((3,3), np.uint8)
            roi_binary = cv2.morphologyEx(roi_binary, cv2.MORPH_OPEN, kernel)
            roi_binary = cv2.morphologyEx(roi_binary, cv2.MORPH_CLOSE, kernel)
            
            # Tìm contours
            contours, _ = cv2.findContours(roi_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                self.show_message("Cảnh báo", "Không tìm thấy vùng ROI phù hợp!", "warning")
                return
            
            # Lọc contours theo diện tích và hình dạng
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Loại bỏ các vùng quá nhỏ (text, nhiễu)
                if area < min_area:
                    continue
                
                # Tính tỷ lệ khung hình để loại bỏ các vùng quá dài/rộng
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / min(w, h)
                
                # Chỉ giữ các vùng có tỷ lệ hợp lý (không quá dài/rộng)
                if aspect_ratio < 5.0:  # Điều chỉnh theo nhu cầu
                    # Tính circularity để ưu tiên các hình tròn/oval
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        if circularity > 0.1:  # Loại bỏ các hình quá méo
                            valid_contours.append((contour, area, circularity))
            
            if not valid_contours:
                self.show_message("Cảnh báo", "Không tìm thấy vùng ROI hợp lệ!", "warning")
                return
            
            # Sắp xếp theo diện tích và circularity
            valid_contours.sort(key=lambda x: (x[1] * x[2]), reverse=True)
            
            # Tạo mask ROI từ contour tốt nhất
            self.roi_mask = np.zeros_like(self.image_data)
            
            # Sử dụng contour lớn nhất và tròn nhất
            best_contour = valid_contours[0][0]
            cv2.fillPoly(self.roi_mask, [best_contour], 255)
            
            # Mở rộng ROI một chút để không bỏ sót
            kernel_expand = np.ones((5,5), np.uint8)
            self.roi_mask = cv2.dilate(self.roi_mask, kernel_expand, iterations=2)
            
            # Áp dụng ROI mask lên ảnh gốc
            roi_image = cv2.bitwise_and(self.image_data, self.roi_mask)
            
            # Hiển thị ROI
            self.display_image(self.label_roi, roi_image)
            
            # Cập nhật trạng thái
            self.btn_process.setEnabled(True)
            
            # Thống kê ROI
            roi_area = np.sum(self.roi_mask > 0)
            total_area = self.roi_mask.shape[0] * self.roi_mask.shape[1]
            
            self.text_result.append(f"✅ Đã phát hiện ROI:")
            self.text_result.append(f"📏 Diện tích ROI: {roi_area:,} pixels")
            self.text_result.append(f"📊 Tỷ lệ ROI: {roi_area/total_area*100:.1f}%")
            self.text_result.append(f"🔍 Số contours hợp lệ: {len(valid_contours)}")
            
            # Hiển thị thông tin contour tốt nhất
            best_area = valid_contours[0][1]
            best_circularity = valid_contours[0][2]
            x, y, w, h = cv2.boundingRect(best_contour)
            
            self.text_result.append(f"🎯 ROI được chọn:")
            self.text_result.append(f"  📏 Diện tích: {best_area:.0f}")
            self.text_result.append(f"  🔘 Circularity: {best_circularity:.3f}")
            self.text_result.append(f"  📐 Bounding box: {w}x{h}")
            self.text_result.append(f"  📍 Vị trí: ({x},{y})")
            
        except Exception as e:
            self.show_message("Lỗi", f"Lỗi khi phát hiện ROI!\nLỗi: {str(e)}", "error")

    def preprocess_image(self):
        """Tiền xử lý ảnh với ROI mask"""
        if self.image_data is None:
            self.show_message("Cảnh báo", "Chưa tải ảnh!", "warning")
            return

        try:
            # Sử dụng ROI nếu có
            if self.roi_mask is not None:
                # Chỉ xử lý trong vùng ROI
                img_roi = cv2.bitwise_and(self.image_data, self.roi_mask)
                self.text_result.append(f"\n🔧 Tiền xử lý trong vùng ROI...")
            else:
                # Xử lý toàn bộ ảnh nếu không có ROI
                img_roi = self.image_data.copy()
                self.text_result.append(f"\n🔧 Tiền xử lý toàn bộ ảnh...")
            
            # Lọc nhiễu
            img_blur = cv2.GaussianBlur(img_roi, (3, 3), 0)
            
            # Phân ngưỡng với adaptive threshold để tốt hơn
            img_thresh = cv2.adaptiveThreshold(
                img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Nếu có ROI mask, chỉ giữ lại pixel trong ROI
            if self.roi_mask is not None:
                img_thresh = cv2.bitwise_and(img_thresh, self.roi_mask)
            
            # Loại bỏ các vùng nhỏ (nhiễu)
            kernel = np.ones((2,2), np.uint8)
            img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
            
            # Loại bỏ các connected components quá nhỏ
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_thresh, connectivity=8)
            
            # Tạo ảnh sạch chỉ giữ lại các component đủ lớn
            min_component_size = 20  # Điều chỉnh theo nhu cầu
            img_clean = np.zeros_like(img_thresh)
            
            for i in range(1, num_labels):  # Bỏ qua background (label 0)
                if stats[i, cv2.CC_STAT_AREA] >= min_component_size:
                    img_clean[labels == i] = 255
            
            self.image_binary = img_clean
            self.display_image(self.label_preprocessed, img_clean)
            
            # Cập nhật trạng thái
            self.btn_dbscan.setEnabled(True)
            
            # Thống kê sau tiền xử lý
            white_pixels = np.sum(img_clean == 255)
            total_pixels = img_clean.shape[0] * img_clean.shape[1]
            
            if self.roi_mask is not None:
                roi_pixels = np.sum(self.roi_mask > 0)
                white_ratio_in_roi = white_pixels / roi_pixels * 100 if roi_pixels > 0 else 0
                self.text_result.append(f"✅ Tiền xử lý hoàn thành (trong ROI):")
                self.text_result.append(f"⚪ Pixel trắng: {white_pixels:,}")
                self.text_result.append(f"📊 Tỷ lệ trong ROI: {white_ratio_in_roi:.2f}%")
            else:
                self.text_result.append(f"✅ Tiền xử lý hoàn thành:")
                self.text_result.append(f"⚪ Pixel trắng: {white_pixels:,}")
                self.text_result.append(f"📊 Tỷ lệ: {white_pixels/total_pixels*100:.2f}%")
            
            self.text_result.append(f"🧹 Đã loại bỏ {num_labels-1} components, giữ lại các components ≥ {min_component_size} pixels")
            
        except Exception as e:
            self.show_message("Lỗi", f"Lỗi khi tiền xử lý!\nLỗi: {str(e)}", "error")

    def run_dbscan(self):
        """Chạy thuật toán DBSCAN"""
        if self.image_binary is None:
            self.show_message("Cảnh báo", "Chưa tiền xử lý ảnh!", "warning")
            return

        try:
            eps = self.slider_eps.value()
            min_pts = self.spin_minpts.value()
            
            # Lấy tọa độ các pixel trắng
            coords = np.column_stack(np.where(self.image_binary > 0))
            
            if coords.shape[0] == 0:
                self.show_message("Cảnh báo", "Không tìm thấy pixel trắng nào để phân tích!", "warning")
                return
            
            self.coords = coords
            
            # Chạy DBSCAN
            self.text_result.append(f"\n🔍 Bắt đầu phân tích DBSCAN...")
            self.text_result.append(f"📍 Số điểm cần phân tích: {coords.shape[0]:,}")
            self.text_result.append(f"⚙️ Tham số: eps={eps}, min_samples={min_pts}")
            
            clustering = DBSCAN(eps=eps, min_samples=min_pts).fit(coords)
            self.clustering_labels = clustering.labels_
            
            # Tạo ảnh kết quả
            self.result_img = self.create_result_image()
            self.display_image(self.label_result, self.result_img)
            
            # Phân tích kết quả
            self.analyze_clusters()
            
            # Cập nhật trạng thái
            self.btn_save_result.setEnabled(True)
            self.btn_save_color.setEnabled(True)
            
        except Exception as e:
            self.show_message("Lỗi", f"Lỗi khi phân tích DBSCAN!\nLỗi: {str(e)}", "error")

    def create_result_image(self):
        """Tạo ảnh kết quả với các cụm được đánh dấu"""
        if self.clustering_labels is None or self.coords is None:
            return None
            
        # Tạo ảnh màu để hiển thị kết quả
        height, width = self.original_image.shape
        result_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Hiển thị ảnh gốc làm nền
        result_img[:, :, 0] = self.original_image
        result_img[:, :, 1] = self.original_image
        result_img[:, :, 2] = self.original_image
        
        # Định nghĩa màu sắc cho các cụm
        colors = [
            (255, 0, 0),    # Đỏ
            (0, 255, 0),    # Xanh lá
            (0, 0, 255),    # Xanh dương
            (255, 255, 0),  # Vàng
            (255, 0, 255),  # Tím
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Cam
            (128, 0, 255),  # Tím đậm
            (255, 192, 203), # Hồng
            (128, 128, 128), # Xám
        ]
        
        # Tô màu các cụm và vẽ bounding box
        unique_labels = sorted([label for label in set(self.clustering_labels) if label != -1])
        
        for i, label in enumerate(unique_labels):
            cluster_coords = self.coords[self.clustering_labels == label]
            color = colors[i % len(colors)]
            
            # Tô màu các pixel trong cụm
            for x, y in cluster_coords:
                if 0 <= x < height and 0 <= y < width:
                    result_img[x, y] = color
            
            # Vẽ bounding box
            min_x, max_x = np.min(cluster_coords[:, 0]), np.max(cluster_coords[:, 0])
            min_y, max_y = np.min(cluster_coords[:, 1]), np.max(cluster_coords[:, 1])
            
            # Vẽ khung bao quanh cụm
            cv2.rectangle(result_img, (min_y-2, min_x-2), (max_y+2, max_x+2), color, 2)
            
            # Vẽ số thứ tự cụm
            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2
            
            # Vẽ nền trắng cho số
            cv2.circle(result_img, (center_y, center_x), 15, (255, 255, 255), -1)
            cv2.circle(result_img, (center_y, center_x), 15, color, 2)
            
            # Vẽ số thứ tự
            cv2.putText(result_img, str(i+1), (center_y-8, center_x+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Đánh dấu các điểm nhiễu bằng màu đen
        noise_coords = self.coords[self.clustering_labels == -1]
        for x, y in noise_coords:
            if 0 <= x < height and 0 <= y < width:
                result_img[x, y] = (64, 64, 64)  # Xám đậm cho noise
        
        # Chuyển về grayscale cho hiển thị
        result_gray = cv2.cvtColor(result_img, cv2.COLOR_RGB2GRAY)
        
        # Lưu ảnh màu để có thể lưu sau này
        self.result_img_color = result_img.copy()
        
        return result_gray

    def analyze_clusters(self):
        """Phân tích chi tiết các cụm"""
        if self.clustering_labels is None or self.coords is None:
            return
            
        unique_labels = set(self.clustering_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(self.clustering_labels).count(-1)
        
        self.text_result.append(f"\n📊 KẾT QUẢ PHÂN TÍCH:")
        self.text_result.append(f"🔢 Số cụm tìm được: {n_clusters}")
        self.text_result.append(f"🔇 Điểm nhiễu: {n_noise}")
        self.text_result.append(f"📍 Tổng điểm: {len(self.clustering_labels)}")
        
        if self.roi_mask is not None:
            self.text_result.append(f"🎯 Phân tích trong vùng ROI")
        
        if n_clusters == 0:
            self.text_result.append("⚠️ Không tìm thấy cụm nào. Hãy thử điều chỉnh tham số.")
            return
        
        self.text_result.append(f"\n{'='*50}")
        self.text_result.append("CHI TIẾT CÁC CỤM:")
        
        height, width = self.original_image.shape
        
        # Phân loại cụm theo khả năng vôi hóa
        high_calcification = []
        medium_calcification = []
        low_calcification = []
        
        for label in sorted(unique_labels):
            if label == -1:
                continue
                
            cluster_coords = self.coords[self.clustering_labels == label]
            
            # Tính toán thống kê
            cluster_size = len(cluster_coords)
            
            # Tính giá trị HU trung bình
            hu_values = []
            for x, y in cluster_coords:
                if 0 <= x < height and 0 <= y < width:
                    hu_values.append(int(self.original_image[x, y]))
            
            if hu_values:
                mean_hu = np.mean(hu_values)
                std_hu = np.std(hu_values)
                min_hu = np.min(hu_values)
                max_hu = np.max(hu_values)
            else:
                mean_hu = std_hu = min_hu = max_hu = 0
            
            # Tính circularity và compactness
            circularity = self.calculate_circularity(cluster_coords)
            compactness = self.calculate_compactness(cluster_coords)
            
            # Tính bounding box
            min_x, max_x = np.min(cluster_coords[:, 0]), np.max(cluster_coords[:, 0])
            min_y, max_y = np.min(cluster_coords[:, 1]), np.max(cluster_coords[:, 1])
            bbox_width = max_y - min_y + 1
            bbox_height = max_x - min_x + 1
            
            # Tính diện tích thực vs diện tích bounding box
            area_ratio = cluster_size / (bbox_width * bbox_height)
            
            # Hiển thị thông tin cụm
            self.text_result.append(f"\n🎯 CỤM #{label + 1}:")
            self.text_result.append(f"  📏 Kích thước: {cluster_size} pixels")
            self.text_result.append(f"  📊 HU trung bình: {mean_hu:.1f} ± {std_hu:.1f}")
            self.text_result.append(f"  📈 HU min-max: {min_hu} - {max_hu}")
            self.text_result.append(f"  🔘 Circularity: {circularity:.3f}")
            self.text_result.append(f"  📦 Compactness: {compactness:.3f}")
            self.text_result.append(f"  📐 Bounding box: {bbox_width}x{bbox_height}")
            self.text_result.append(f"  📍 Vị trí: ({min_x},{min_y}) - ({max_x},{max_y})")
            self.text_result.append(f"  🎯 Area ratio: {area_ratio:.3f}")
            
            # Đánh giá khả năng vôi hóa dựa trên nhiều tiêu chí
            calcification_score = self.calculate_calcification_score(
                mean_hu, cluster_size, circularity, compactness, area_ratio
            )
            
            if calcification_score >= 0.7:
                confidence = "CAO"
                confidence_color = "🔴"
                high_calcification.append(label + 1)
            elif calcification_score >= 0.4:
                confidence = "TRUNG BÌNH" 
                confidence_color = "🟡"
                medium_calcification.append(label + 1)
            else:
                confidence = "THẤP"
                confidence_color = "🟢"
                low_calcification.append(label + 1)
            
            self.text_result.append(f"  {confidence_color} Khả năng vôi hóa: {confidence} (Score: {calcification_score:.2f})")
        
        # Tổng kết đánh giá
        self.text_result.append(f"\n{'='*50}")
        self.text_result.append("TỔNG KẾT ĐÁNH GIÁ:")
        
        if high_calcification:
            self.text_result.append(f"🔴 Nguy cơ cao ({len(high_calcification)} cụm): {high_calcification}")
        
        if medium_calcification:
            self.text_result.append(f"🟡 Nguy cơ trung bình ({len(medium_calcification)} cụm): {medium_calcification}")
        
        if low_calcification:
            self.text_result.append(f"🟢 Nguy cơ thấp ({len(low_calcification)} cụm): {low_calcification}")
        
        # Khuyến nghị
        if high_calcification:
            self.text_result.append(f"\n⚠️ KHUYẾN NGHỊ: Phát hiện {len(high_calcification)} vùng nguy cơ cao.")
            self.text_result.append("   Cần theo dõi và tham khảo ý kiến chuyên gia.")
        elif medium_calcification:
            self.text_result.append(f"\n💡 KHUYẾN NGHỊ: Phát hiện {len(medium_calcification)} vùng nguy cơ trung bình.")
            self.text_result.append("   Nên theo dõi định kỳ.")
        else:
            self.text_result.append("\n✅ ĐÁNH GIÁ: Các vùng phát hiện có nguy cơ thấp.")

    def calculate_calcification_score(self, mean_hu, size, circularity, compactness, area_ratio):
        """Tính điểm khả năng vôi hóa dựa trên nhiều tiêu chí"""
        score = 0.0
        
        # Điểm dựa trên giá trị HU (40% trọng số)
        if mean_hu > 150:
            hu_score = 1.0
        elif mean_hu > 120:
            hu_score = 0.8
        elif mean_hu > 100:
            hu_score = 0.6
        elif mean_hu > 80:
            hu_score = 0.4
        else:
            hu_score = 0.2
        
        score += hu_score * 0.4
        
        # Điểm dựa trên kích thước (20% trọng số)
        if size > 100:
            size_score = 1.0
        elif size > 50:
            size_score = 0.8
        elif size > 25:
            size_score = 0.6
        else:
            size_score = 0.3
        
        score += size_score * 0.2
        
        # Điểm dựa trên hình dạng (25% trọng số)
        shape_score = (circularity + compactness) / 2
        score += shape_score * 0.25
        
        # Điểm dựa trên mật độ (15% trọng số)
        density_score = min(area_ratio * 2, 1.0)  # Chuẩn hóa về [0,1]
        score += density_score * 0.15
        
        return min(score, 1.0)

    def calculate_circularity(self, coords):
        """Tính độ tròn của cụm"""
        if len(coords) < 3:
            return 0
        
        try:
            # Tạo mask từ coordinates
            min_x, max_x = np.min(coords[:, 0]), np.max(coords[:, 0])
            min_y, max_y = np.min(coords[:, 1]), np.max(coords[:, 1])
            
            mask = np.zeros((max_x - min_x + 1, max_y - min_y + 1), dtype=np.uint8)
            
            for x, y in coords:
                mask[x - min_x, y - min_y] = 255
            
            # Tìm contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                cnt = contours[0]
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    return min(circularity, 1.0)  # Giới hạn tối đa = 1
            
            return 0
            
        except Exception:
            return 0

    def calculate_compactness(self, coords):
        """Tính độ compact của cụm"""
        if len(coords) < 2:
            return 0
        
        try:
            # Tính trung tâm mass
            center_x = np.mean(coords[:, 0])
            center_y = np.mean(coords[:, 1])
            
            # Tính khoảng cách trung bình từ các điểm đến trung tâm
            distances = []
            for x, y in coords:
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                distances.append(dist)
            
            if not distances:
                return 0
            
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            
            # Compactness = 1 - (std/mean), giá trị cao = compact hơn
            if mean_dist > 0:
                compactness = max(0, 1 - (std_dist / mean_dist))
                return min(compactness, 1.0)
            
            return 0
            
        except Exception:
            return 0

    def save_result_image(self):
        """Lưu ảnh kết quả"""
        if self.result_img is None:
            self.show_message("Cảnh báo", "Chưa có kết quả để lưu!", "warning")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Lưu ảnh kết quả", "dbscan_roi_result.png", 
            "Images (*.png *.jpg *.bmp)"
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.result_img)
                self.show_message("Thành công", f"Đã lưu ảnh kết quả tại:\n{file_path}")
                
                # Lưu thêm báo cáo text
                report_path = file_path.replace('.png', '_report.txt').replace('.jpg', '_report.txt').replace('.bmp', '_report.txt')
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(self.text_result.toPlainText())
                
                self.text_result.append(f"\n💾 Đã lưu báo cáo tại: {report_path}")
                
            except Exception as e:
                self.show_message("Lỗi", f"Không thể lưu file!\nLỗi: {str(e)}", "error")

    def save_color_image(self):
        """Lưu ảnh màu với các cụm được đánh dấu"""
        if self.result_img_color is None:
            self.show_message("Cảnh báo", "Chưa có kết quả màu để lưu!", "warning")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Lưu ảnh màu", "dbscan_roi_color_result.png", 
            "Images (*.png *.jpg *.bmp)"
        )
        
        if file_path:
            try:
                # Chuyển từ RGB sang BGR cho OpenCV
                img_bgr = cv2.cvtColor(self.result_img_color, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, img_bgr)
                self.show_message("Thành công", f"Đã lưu ảnh màu tại:\n{file_path}")
                
                # Tạo legend cho các màu
                legend_path = file_path.replace('.png', '_legend.txt').replace('.jpg', '_legend.txt').replace('.bmp', '_legend.txt')
                self.create_color_legend(legend_path)
                
                self.text_result.append(f"\n🎨 Đã lưu ảnh màu tại: {file_path}")
                self.text_result.append(f"📋 Đã lưu chú thích màu tại: {legend_path}")
                
            except Exception as e:
                self.show_message("Lỗi", f"Không thể lưu file màu!\nLỗi: {str(e)}", "error")

    def create_color_legend(self, legend_path):
        """Tạo file chú thích màu sắc"""
        colors = [
            "Đỏ", "Xanh lá", "Xanh dương", "Vàng", "Tím",
            "Cyan", "Cam", "Tím đậm", "Hồng", "Xám"
        ]
        
        try:
            with open(legend_path, 'w', encoding='utf-8') as f:
                f.write("CHƯƠNG TRÌNH PHÂN TÍCH VÔI HÓA MẠCH MÁU - DBSCAN với ROI\n")
                f.write("="*60 + "\n\n")
                f.write("CHÚ THÍCH MÀU SẮC CÁC CỤM:\n")
                f.write("-" * 30 + "\n")
                
                unique_labels = sorted([label for label in set(self.clustering_labels) if label != -1])
                
                for i, label in enumerate(unique_labels):
                    cluster_coords = self.coords[self.clustering_labels == label]
                    cluster_size = len(cluster_coords)
                    color_name = colors[i % len(colors)]
                    
                    # Tính điểm khả năng vôi hóa cho legend
                    hu_values = [int(self.original_image[x, y]) for x, y in cluster_coords 
                                if 0 <= x < self.original_image.shape[0] and 0 <= y < self.original_image.shape[1]]
                    
                    if hu_values:
                        mean_hu = np.mean(hu_values)
                        circularity = self.calculate_circularity(cluster_coords)
                        compactness = self.calculate_compactness(cluster_coords)
                        
                        min_x, max_x = np.min(cluster_coords[:, 0]), np.max(cluster_coords[:, 0])
                        min_y, max_y = np.min(cluster_coords[:, 1]), np.max(cluster_coords[:, 1])
                        bbox_area = (max_x - min_x + 1) * (max_y - min_y + 1)
                        area_ratio = cluster_size / bbox_area if bbox_area > 0 else 0
                        
                        calc_score = self.calculate_calcification_score(
                            mean_hu, cluster_size, circularity, compactness, area_ratio
                        )
                        
                        if calc_score >= 0.7:
                            risk_level = "CAO"
                        elif calc_score >= 0.4:
                            risk_level = "TRUNG BÌNH"
                        else:
                            risk_level = "THẤP"
                        
                        f.write(f"Cụm #{i+1}: {color_name} - {cluster_size} pixels - Nguy cơ: {risk_level}\n")
                        f.write(f"    HU trung bình: {mean_hu:.1f}, Score: {calc_score:.2f}\n")
                
                f.write(f"\nĐiểm nhiễu: Xám đậm\n")
                f.write(f"Khung viền: Bao quanh mỗi cụm\n")
                f.write(f"Số thứ tự: Hiển thị ở giữa mỗi cụm\n\n")
                
                if self.roi_mask is not None:
                    f.write("LƯU Ý: Phân tích được thực hiện trong vùng ROI được phát hiện tự động\n")
                    f.write("để loại bỏ nhiễu từ text và các thông tin ngoài hình ảnh chính.\n")
                
        except Exception as e:
            print(f"Không thể tạo file chú thích: {e}")

    def display_image(self, target_label, img):
        """Hiển thị ảnh lên label"""
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
    app.setStyle('Fusion')  # Giao diện đẹp hơn
    
    window = DBSCANApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()