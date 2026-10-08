import os

# 项目根目录
ROOT_PATH = os.path.dirname(
    os.path.dirname(__file__)
)

# 数据路径
DATA_PATH = os.path.join(
    ROOT_PATH,
    "data"
)

# 测试图片路径
TEST_PATH = os.path.join(
    ROOT_PATH,
    "testimg"
)

# 模型路径
MODEL_PATH = os.path.join(
    ROOT_PATH,
    "models",
    "vehicle_cnn.h5"
)

# 结果保存路径
RESULT_PATH = os.path.join(
    ROOT_PATH,
    "results"
)

# =========================
# 模型相关配置
# =========================

BACKBONE = "CustomCNN"

# 迁移学习标准输入尺寸（正方形）；192 在速度与精度间平衡
IMG_SIZE = 192

# 兼容旧脚本的别名
IMG_HEIGHT = IMG_SIZE
IMG_WIDTH = IMG_SIZE

# 类别数量
NUM_CLASSES = 10

# 数据划分比例: 训练/验证/测试
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# batch训练次数
BATCH_SIZE = 32

# 随机种子（保证可复现）
SEED = 42

# 两阶段训练轮数
HEAD_EPOCHS = 30       # 阶段一：冻结骨干，仅训练分类头
FINETUNE_EPOCHS = 12    # 阶段二：解冻部分骨干，微调

# 学习率
HEAD_LEARNING_RATE = 1e-3
FINETUNE_LEARNING_RATE = 1e-5

# 微调时解冻的骨干层数（从末尾倒数）
FINETUNE_AT = 30

# Label Smoothing，缓解过拟合/过自信
LABEL_SMOOTHING = 0.1

# 分类头丢弃率
DROPOUT_RATE = 0.5
