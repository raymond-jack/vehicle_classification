import os
import json
import numpy as np
import cv2
import tensorflow as tf

from utils.config import (
    MODEL_PATH,
    IMG_HEIGHT,
    IMG_WIDTH,
)

# 类别名文件相对路径（与训练脚本一致）
CLASS_NAMES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "utils",
    "class_names.json",
)


class VehiclePredictor:
    """车辆分类推理器：加载模型，对单张图片/摄像头帧做预测。"""

    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        # 关闭 TF 冗余日志
        tf.get_logger().setLevel("ERROR")
        self.model = tf.keras.models.load_model(self.model_path)
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            self.class_names = json.load(f)
        self.num_classes = len(self.class_names)

    def preprocess(self, image_bgr):
        """
        预处理（与训练保持一致）：
          BGR -> RGB -> resize -> /255.0 -> 扩维
        模型输入为 [0,1] 浮点，迁移学习分支内部再做 ImageNet 归一化。
        """
        image = cv2.resize(image_bgr, (IMG_WIDTH, IMG_HEIGHT))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    def predict(self, image_path):
        """按路径预测单张图片"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        return self.predict_frame(image)

    def predict_frame(self, frame):
        """对一帧（BGR numpy 数组）做预测，返回 (label, confidence)"""
        x = self.preprocess(frame)
        pred = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(pred))
        return self.class_names[idx], float(pred[idx])

    def predict_topk(self, frame, k=3):
        """返回 Top-K 预测列表 [(label, confidence), ...]（降序）"""
        x = self.preprocess(frame)
        pred = self.model.predict(x, verbose=0)[0]
        order = np.argsort(pred)[::-1][:k]
        return [(self.class_names[i], float(pred[i])) for i in order]
