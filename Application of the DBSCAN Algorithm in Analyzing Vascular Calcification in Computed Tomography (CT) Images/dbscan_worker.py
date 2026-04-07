from PyQt6.QtCore import QThread, pyqtSignal
from sklearn.cluster import DBSCAN
import numpy as np

class DBSCANWorker(QThread):
    finished = pyqtSignal(object)  # Emit labels
    error = pyqtSignal(str)

    def __init__(self, coords, eps, min_points):
        super().__init__()
        self.coords = coords
        self.eps = eps
        self.min_points = min_points

    def run(self):
        try:
            if self.coords is None or self.coords.shape[0] == 0:
                self.error.emit("Không có dữ liệu điểm để phân tích.")
                return
                
            # Chạy thuật toán tốn thời gian ở đây
            clustering = DBSCAN(eps=self.eps, min_samples=self.min_points).fit(self.coords)
            
            # Gửi kết quả về thread chính của GUI
            self.finished.emit(clustering.labels_)
        except Exception as e:
            self.error.emit(f"Lỗi khối DBSCAN Backend: {str(e)}")
