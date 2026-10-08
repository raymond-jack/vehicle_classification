# 车辆图像分类识别系统 (Vehicle Classification)

基于深度学习的端到端车辆图像分类识别系统，支持 **10 类车辆**图片识别与**摄像头实时识别**，
提供 **PyQt5 桌面端**与 **Gradio 网页端**两套交互界面。

> 个人独立开发项目：从数据处理、模型设计训练、评估可视化到多端部署全部自行实现。

---

## ✨ 功能特性

- **10 类车辆识别**：SUV、bus、family sedan、fire engine、heavy truck、jeep、minibus、racing car、taxi、truck
- **自研 CNN 模型**：离线可用，无需联网下载预训练权重
- **完整训练流程**：数据增强、标签平滑、自动回调（ModelCheckpoint / EarlyStopping / ReduceLROnPlateau）
- **可视化评估**：准确率/损失曲线、混淆矩阵热力图、数据集分布统计
- **双端部署**：PyQt5 桌面客户端 + Gradio 网页端，共享同一推理核心
- **实时识别**：支持摄像头画面识别，输出 Top-3 类别与置信度

---

## 🧱 技术栈

| 类别 | 技术 |
|---|---|
| 语言 | Python 3.12 |
| 深度学习 | TensorFlow / Keras |
| 图像处理 | OpenCV、Pillow |
| 数值与可视化 | NumPy、Matplotlib、scikit-learn、seaborn |
| 界面 / 部署 | PyQt5（桌面端）、Gradio（网页端） |

---

## 📁 项目结构

```
vehicle_classification/
├── data/                      # 数据集（10 个子目录，每类需自行准备图片）
├── testimg/                   # 单张测试图片示例
├── models/
│   └── vehicle_cnn.h5         # 训练好的最佳模型权重
├── results/                   # 训练与评估结果（图片 + JSON）
├── utils/
│   ├── config.py              # 全局配置（路径、超参数）
│   └── class_names.json       # 类别名称映射
├── app/
│   ├── vehicle_predictor.py   # 推理封装（统一入口）
│   ├── camera.py              # 摄像头管理
│   └── gui.py                 # PyQt5 桌面界面
├── 01-dataset_analysis.py     # 数据集分析与可视化
├── 02_data_preprocessing.py   # 数据加载 / 预处理 / 增强 / 划分
├── 03_model_build.py          # 模型结构定义
├── 04_train.py                # 模型训练入口
├── 05_evaluate.py             # 模型评估与可视化
├── 06_predict.py              # 单图预测示例
├── main.py                    # PyQt5 桌面端启动
└── Gradio_main.py             # Gradio 网页端启动
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据集

按类别在 `data/` 下建立子目录，每个类别一个文件夹：

```
data/
├── SUV/
├── bus/
├── family sedan/
└── ...
```

> 本仓库未附带数据集，请自行准备（本项目使用 10 类 × 300 张 = 3000 张均衡数据）。

### 3. 运行流程

```bash
python 01-dataset_analysis.py    # 数据分析与可视化
python 02_data_preprocessing.py  # 数据预处理与划分
python 04_train.py               # 模型训练
python 05_evaluate.py            # 模型评估
python 06_predict.py             # 单图预测示例
```

### 4. 启动界面

```bash
python main.py           # PyQt5 桌面端
python Gradio_main.py    # Gradio 网页端 -> http://localhost:7860
```

---

## 🧠 模型设计

```
输入 192×192×3
  → Conv2D(32)  + ReLU + MaxPool
  → Conv2D(64)  + ReLU + MaxPool
  → Conv2D(128) + ReLU + MaxPool + Dropout(0.25)
  → Conv2D(256) + ReLU + MaxPool + Dropout(0.25)
  → Conv2D(512) + ReLU + MaxPool + Dropout(0.25)
  → GlobalAveragePooling2D
  → Dense(256, relu) + Dropout(0.5)
  → Dense(10, softmax)
```

设计要点：

- **通道数递进** 32 → 64 → 128 → 256 → 512，逐步提取从低级纹理到高级语义的特征
- **He 正态初始化**，适配 ReLU 激活，缓解梯度消失
- **全局平均池化（GAP）**替代 Flatten，大幅减少参数量、抑制过拟合
- **双重 Dropout**：高通道卷积层 0.25 + 分类头 0.5
- **数据增强**（仅训练集）：随机水平翻转 + 亮度 / 对比度 / 饱和度 / 色相轻度扰动
- **Label Smoothing** 缓解模型过自信

训练策略：固定随机种子保证可复现，使用 ModelCheckpoint 保存验证集最优权重、
EarlyStopping 自动早停、ReduceLROnPlateau 自适应衰减学习率。

---

## 📊 实验结果

在 10 类、3000 张均衡车辆图像（训练/验证/测试 = 70%/15%/15%）上的测试结果：

| 指标 | 数值 |
|---|---|
| 测试准确率 | **90.00%** |
| 测试损失 | 0.4651 |

混淆矩阵显示 SUV、bus、fire engine、truck 等类别识别准确率接近 100%；
误分类主要集中在外观相近的类别之间（heavy truck ↔ truck、family sedan ↔ taxi ↔ jeep、minibus ↔ truck）。

结果可视化见 `results/` 目录：

| 文件 | 说明 |
|---|---|
| `dataset_pie.png` / `dataset_bar.png` | 数据集类别分布 |
| `dataset_samples.png` | 各类别样本示例 |
| `accuracy.png` / `loss.png` | 训练与验证曲线 |
| `confusion_matrix.png` | 测试集混淆矩阵 |
| `gui_pyqt5_layout.png` / `gui_gradio_layout.png` | 界面预览 |

---

## 🔌 推理接口

所有界面共用 `app/vehicle_predictor.py` 中的 `VehiclePredictor`，保证训练与推理预处理严格一致：

```python
from app.vehicle_predictor import VehiclePredictor

predictor = VehiclePredictor()
label, confidence = predictor.predict("testimg/bus_test.jpg")
top3 = predictor.predict_topk(frame, k=3)   # [(label, confidence), ...]
```

---

## 📄 License

本项目仅供学习与交流使用。
