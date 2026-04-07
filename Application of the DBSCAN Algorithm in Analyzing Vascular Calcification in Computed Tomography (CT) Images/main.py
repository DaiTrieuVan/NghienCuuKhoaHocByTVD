import sys
import os
import numpy as np
import cv2
from PyQt6.QtWidgets import (
    QApplication, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QHBoxLayout, QSlider, QSpinBox,
    QTextEdit, QFrame, QMessageBox, QProgressDialog
)
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import Qt

from utils import load_image_file, preprocess_image_pipeline, calculate_circularity
from dbscan_worker import DBSCANWorker

class DBSCANApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Phân tích vôi hóa mạch máu - DBV2 (Tối ưu hóa)")
        self.setGeometry(100, 100, 1600, 800)

        # Biến lưu trữ
        self.image_path = None
        self.original_hu_matrix = None   # Mảng thật (có HU với DICOM) dùng để định lượng
        self.display_image = None        # Mảng 0-255 dùng làm GUI gốc
        self.image_data = None           # grayscale (0-255) version for display
        self.image_binary = None         # nhị phân, same size as original
        self.result_img = None
        self.result_img_color = None
        self.clustering_labels = None
        self.coords = None
        self.crop_offset = (0, 0)

        self.worker = None

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

        param_title = QLabel("Tham số DBSCAN & Chẩn đoán")
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
        
        hu_layout = QHBoxLayout()
        hu_layout.addWidget(QLabel("Ngưỡng vôi hoá (HU):"))
        self.spin_hu_threshold = QSpinBox()
        self.spin_hu_threshold.setMinimum(-1000)
        self.spin_hu_threshold.setMaximum(3000)
        self.spin_hu_threshold.setValue(130)
        hu_layout.addWidget(self.spin_hu_threshold)
        param_layout.addLayout(hu_layout)

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
        self.btn_dbscan.clicked.connect(self.start_dbscan)
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
            hu_matrix, display_image = load_image_file(file_path)

            self.original_hu_matrix = hu_matrix.copy()
            self.display_image = display_image.copy()
            self.image_data = None
            self.image_binary = None
            self.result_img = None
            self.result_img_color = None
            self.clustering_labels = None
            self.coords = None
            self.crop_offset = (0, 0)

            self.display_on_label(self.label_original, display_image)
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
            self.text_result.append(f"Kích thước: {display_image.shape[1]} x {display_image.shape[0]} pixels")
            self.text_result.append(f"Giá trị HU (thật): {np.min(hu_matrix):.1f} - {np.max(hu_matrix):.1f}")
            self.text_result.append("Chú ý: Dữ liệu HU đã được bảo toàn thông qua hàm Slope/Intercept để phân tích chính xác.\n")

        except Exception as e:
            self.show_message("Lỗi", f"Không thể đọc ảnh!\nLỗi: {str(e)}", "error")

    def preprocess_image(self):
        if self.display_image is None:
            self.show_message("Cảnh báo", "Chưa tải ảnh!", "warning")
            return
        try:
            img_bin, img_data, overlay, offset, bbox = preprocess_image_pipeline(self.display_image)
            self.image_binary = img_bin
            # self.image_data will be background for displaying clustering results
            self.image_data = img_data
            self.crop_offset = offset
            
            self.display_on_label(self.label_preprocessed, overlay)
            self.btn_dbscan.setEnabled(True)
            
            white_pixels = np.sum(self.image_binary == 255)
            total_pixels = self.image_binary.shape[0] * self.image_binary.shape[1]
            
            y0, y1, x0, x1 = bbox
            self.text_result.append("Tiền xử lý hoàn thành (Module utils):")
            self.text_result.append(f"Pixel ứng viên: {white_pixels:,}")
            self.text_result.append(f"Tỷ lệ / Khoảng: {white_pixels/total_pixels*100:.4f}%")
            self.text_result.append(f"Bounding box sọ/ngoại biên: ({x0},{y0}) - ({x1},{y1})\n")

        except Exception as e:
            self.show_message("Lỗi", f"Lỗi khi tiền xử lý!\nLỗi: {str(e)}", "error")

    def start_dbscan(self):
        if self.image_binary is None:
            self.show_message("Cảnh báo", "Chưa tiền xử lý ảnh!", "warning")
            return
            
        eps = self.slider_eps.value()
        min_pts = self.spin_minpts.value()
        coords = np.column_stack(np.where(self.image_binary > 0))
        
        if coords.shape[0] == 0:
            self.show_message("Cảnh báo", "Không tìm thấy pixel trắng nào để phân tích!", "warning")
            return
            
        self.coords = coords
        self.text_result.append(f"Bắt đầu phân tích DBSCAN (Background Thread)...")
        self.text_result.append(f"Số điểm phân tích: {coords.shape[0]:,}")
        self.text_result.append(f"Tham số: eps={eps}, min_samples={min_pts}")
        
        # Disable buttons to wait for the thread
        self.btn_open.setEnabled(False)
        self.btn_process.setEnabled(False)
        self.btn_dbscan.setEnabled(False)
        self.btn_save_result.setEnabled(False)
        self.btn_save_color.setEnabled(False)
        self.slider_eps.setEnabled(False)
        self.spin_minpts.setEnabled(False)
        
        self.worker = DBSCANWorker(coords, eps, min_pts)
        self.worker.finished.connect(self.on_dbscan_finished)
        self.worker.error.connect(self.on_dbscan_error)
        self.worker.start()

    def on_dbscan_finished(self, labels):
        self.clustering_labels = labels
        self.result_img = self.create_result_image()
        self.display_on_label(self.label_result, self.result_img)
        self.analyze_clusters()
        
        # Re-enable UI
        self.btn_open.setEnabled(True)
        self.btn_process.setEnabled(True)
        self.btn_dbscan.setEnabled(True)
        self.btn_save_result.setEnabled(True)
        self.btn_save_color.setEnabled(True)
        self.slider_eps.setEnabled(True)
        self.spin_minpts.setEnabled(True)

    def on_dbscan_error(self, error_message):
        self.show_message("Lỗi DBSCAN", error_message, "error")
        # Re-enable UI
        self.btn_open.setEnabled(True)
        self.btn_process.setEnabled(True)
        self.btn_dbscan.setEnabled(True)
        self.slider_eps.setEnabled(True)
        self.spin_minpts.setEnabled(True)

    def create_result_image(self):
        if self.clustering_labels is None or self.coords is None:
            return None

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
        if self.clustering_labels is None or self.coords is None:
            return

        unique_labels = set(self.clustering_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(self.clustering_labels).count(-1)

        self.text_result.append(f"\nKẾT QUẢ PHÂN TÍCH DBSCAN:")
        self.text_result.append(f"Số cụm tìm được: {n_clusters}")
        self.text_result.append(f"Điểm nhiễu (-1): {n_noise}")

        if n_clusters == 0:
            self.text_result.append("Không tìm thấy cụm. Hãy điều chỉnh tham số Eps/MinPts.")
            return

        self.text_result.append(f"\n{'='*50}")
        self.text_result.append("CHI TIẾT CÁC CỤM (Dựa trên HU thực tế):")

        height, width = self.original_hu_matrix.shape
        hu_threshold = self.spin_hu_threshold.value()

        for label in sorted(unique_labels):
            if label == -1:
                continue

            cluster_coords = self.coords[self.clustering_labels == label]
            cluster_size = len(cluster_coords)

            hu_values = []
            for x, y in cluster_coords:
                if 0 <= x < height and 0 <= y < width:
                    # Tra cứu trực tiếp trên mảng HU vật lý tự nhiên
                    hu_values.append(float(self.original_hu_matrix[x, y]))

            if hu_values:
                mean_hu = np.mean(hu_values)
                std_hu = np.std(hu_values)
                min_hu = np.min(hu_values)
                max_hu = np.max(hu_values)
            else:
                mean_hu = std_hu = min_hu = max_hu = 0

            circularity = calculate_circularity(cluster_coords)

            min_x, max_x = np.min(cluster_coords[:, 0]), np.max(cluster_coords[:, 0])
            min_y, max_y = np.min(cluster_coords[:, 1]), np.max(cluster_coords[:, 1])
            bbox_width = max_y - min_y + 1
            bbox_height = max_x - min_x + 1

            self.text_result.append(f"\nCỤM #{label + 1}:")
            self.text_result.append(f"  Kích thước: {cluster_size} pixels")
            self.text_result.append(f"  HU trung bình: {mean_hu:.1f} ± {std_hu:.1f}")
            self.text_result.append(f"  HU min-max: {min_hu:.1f} - {max_hu:.1f}")
            self.text_result.append(f"  Circularity: {circularity:.3f}")
            self.text_result.append(f"  Bounding box: {bbox_width}x{bbox_height}")

            # Chẩn đoán theo ngưỡng HU Threshold mới
            if mean_hu > hu_threshold:
                confidence = "CAO"
            elif mean_hu > (hu_threshold - 30):
                confidence = "TRUNG BÌNH"
            else:
                confidence = "THẤP"

            self.text_result.append(f"  Khả năng vôi hóa (Ngưỡng {hu_threshold}): {confidence}")

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
            except Exception as e:
                self.show_message("Lỗi", f"Không thể lưu file!\nLỗi: {str(e)}", "error")

    def save_color_image(self):
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
            except Exception as e:
                self.show_message("Lỗi", f"Không thể lưu file màu!\nLỗi: {str(e)}", "error")

    def create_color_legend(self, legend_path):
        colors = ["Đỏ", "Xanh lá", "Xanh dương", "Vàng", "Tím", "Cyan", "Cam", "Tím đậm", "Hồng", "Xám"]
        try:
            with open(legend_path, 'w', encoding='utf-8') as f:
                f.write("CHƯƠNG TRÌNH PHÂN TÍCH VÔI HÓA MẠCH MÁU - DBV2\n==================================================\n\nCHÚ THÍCH MÀU SẮC CÁC CỤM:\n------------------------------\n")
                unique_labels = sorted([label for label in set(self.clustering_labels) if label != -1])
                for i, label in enumerate(unique_labels):
                    cluster_coords = self.coords[self.clustering_labels == label]
                    f.write(f"Cụm #{i+1}: {colors[i % len(colors)]} ({len(cluster_coords)} pixels)\n")
                f.write(f"\nĐiểm nhiễu: Xám đậm\n")
        except Exception as e:
             print(f"Lỗi tạo tệp lưu chú thích màu: {e}")

    def display_on_label(self, target_label, img):
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
