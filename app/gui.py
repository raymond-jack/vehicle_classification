import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox
)
from PyQt5.QtGui import (
    QPixmap,
    QImage
)
from PyQt5.QtCore import (
    Qt,
    QTimer
)
from app.vehicle_predictor import VehiclePredictor
from app.camera import Camera

class VehicleGUI(QWidget):
    """
    车辆分类识别系统GUI
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "车辆图像分类识别系统"
        )
        self.resize(
            900,
            700
        )
        # 模型预测器
        self.predictor = VehiclePredictor()
        # 摄像头对象
        self.camera = None
        # 当前图片
        self.image_path = None
        # 当前摄像头帧
        self.current_frame = None
        # 摄像头预测计数
        self.frame_count = 0
        # 每15帧预测一次
        self.predict_interval = 15
        # 定时器刷新摄像头
        self.timer = QTimer()
        self.timer.timeout.connect(
            self.update_camera_frame
        )
        self.init_ui()
    def init_ui(self):
        """
        初始化界面
        """
        # 图片显示区域
        self.image_label = QLabel()
        self.image_label.setFixedSize(
            640,
            420
        )
        self.image_label.setAlignment(
            Qt.AlignCenter
        )
        self.image_label.setStyleSheet(
            """
            border:2px solid gray;
            """
        )
        # 结果显示
        self.result_label = QLabel(
            "预测结果：等待识别"
        )
        self.result_label.setAlignment(
            Qt.AlignCenter
        )
        self.result_label.setStyleSheet(
            """
            font-size:20px;
            """
        )
        # 图片按钮
        self.open_image_button = QPushButton(
            "选择图片"
        )
        self.predict_button = QPushButton(
            "图片识别"
        )
        self.open_image_button.clicked.connect(
            self.open_image
        )
        self.predict_button.clicked.connect(
            self.predict_image
        )
        # 摄像头选择
        self.camera_box = QComboBox()
        self.camera_box.addItem(
            "摄像头 0"
        )
        self.camera_box.addItem(
            "摄像头 1"
        )
        # 摄像头按钮
        self.open_camera_button = QPushButton(
            "打开摄像头"
        )
        self.close_camera_button = QPushButton(
            "关闭摄像头"
        )
        self.open_camera_button.clicked.connect(
            self.open_camera
        )
        self.close_camera_button.clicked.connect(
            self.close_camera
        )
        # 第一行按钮
        image_layout = QHBoxLayout()
        image_layout.addWidget(
            self.open_image_button
        )
        image_layout.addWidget(
            self.predict_button
        )
        # 第二行按钮
        camera_layout = QHBoxLayout()
        camera_layout.addWidget(
            self.camera_box
        )
        camera_layout.addWidget(
            self.open_camera_button
        )
        camera_layout.addWidget(
            self.close_camera_button
        )
        # 总布局
        layout = QVBoxLayout()
        layout.addWidget(
            self.image_label
        )
        layout.addWidget(
            self.result_label
        )
        layout.addLayout(
            image_layout
        )
        layout.addLayout(
            camera_layout
        )
        self.setLayout(
            layout
        )
    # ============================
    # 图片功能
    # ============================
    def open_image(self):
        """
        选择图片
        """
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择车辆图片",
            "",
            "Images (*.jpg *.png *.jpeg)"
        )
        if path:
            self.image_path = path
            pixmap = QPixmap(
                path
            )
            pixmap = pixmap.scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.KeepAspectRatio
            )
            self.image_label.setPixmap(
                pixmap
            )
    def predict_image(self):
        """
        图片预测
        """
        if self.image_path is None:
            self.result_label.setText(
                "请先选择图片"
            )
            return
        label, confidence = self.predictor.predict(
            self.image_path
        )
        self.result_label.setText(
            f"""
预测类别:
{label}
置信度:
{confidence:.2%}
"""
        )
    # ============================
    # 摄像头功能
    # ============================
    def predict_camera(self):
        """
        摄像头实时车辆预测
        """
        if self.current_frame is None:
            return
        label, confidence = self.predictor.predict_frame(
            self.current_frame
        )
        self.result_label.setText(
            f"""
    实时识别:
    类别:
    {label}
    置信度:
    {confidence:.2%}
    """
        )
    def open_camera(self):
        """
        打开摄像头
        """
        camera_id = self.camera_box.currentIndex()
        self.camera = Camera(
            camera_id
        )
        success = self.camera.open()
        if success:
            self.timer.start(
                30
            )
        else:
            self.result_label.setText(
                "摄像头打开失败"
            )
    def update_camera_frame(self):
        """
        更新摄像头画面 + 实时预测
        """
        if self.camera is None:
            return
        frame = self.camera.read()
        if frame is None:
            return
        # 保存原始BGR帧
        self.current_frame = frame.copy()
        # ======================
        # 实时预测
        # ======================
        self.frame_count += 1
        if self.frame_count % self.predict_interval == 0:
            self.predict_camera()
        # ======================
        # 显示画面
        # ======================
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        image = QImage(
            rgb_frame.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(
            image
        )
        pixmap = pixmap.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.KeepAspectRatio
        )
        self.image_label.setPixmap(
            pixmap
        )
    def close_camera(self):
        """
        关闭摄像头
        """
        if self.timer.isActive():
            self.timer.stop()
        if self.camera:
            self.camera.release()
            self.camera = None
        self.current_frame = None
        self.image_label.clear()
        self.result_label.setText(
            "摄像头已关闭"
        )
    def closeEvent(self,event):
        """
        窗口关闭释放资源
        """
        self.close_camera()
        event.accept()

if __name__ == "__main__":
    app = QApplication(
        sys.argv
    )
    window = VehicleGUI()
    window.show()
    sys.exit(
        app.exec_()
    )