import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix
import seaborn as sns

from utils.config import (
    MODEL_PATH,
    RESULT_PATH,
    DATA_PATH,
    IMG_HEIGHT,
    IMG_WIDTH,
    BATCH_SIZE,
    SEED
)
def load_test_dataset():
    """
    加载验证集作为测试集
    """
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_PATH,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=(
            IMG_HEIGHT,
            IMG_WIDTH
        ),
        batch_size=BATCH_SIZE

    )
    return test_ds

def preprocess_dataset(dataset):
    """
    图片归一化
    """
    normalization = tf.keras.layers.Rescaling(
        1./255
    )
    dataset = dataset.map(
        lambda x,y:
        (
            normalization(x),
            y
        )
    )
    return dataset

def evaluate_model():
    # ======================
    # 1. 加载模型
    # ======================
    model = tf.keras.models.load_model(
        MODEL_PATH
    )
    print(
        "模型加载成功"
    )
    # ======================
    # 2. 加载测试集
    # =====================
    test_ds = load_test_dataset()
    test_ds = preprocess_dataset(
        test_ds
    )
    # ======================
    # 3. 模型评估
    # ======================
    loss, accuracy = model.evaluate(
        test_ds
    )
    print(
        "\n测试集结果"
    )
    print(
        "Loss:",
        loss
    )
    print(
        "Accuracy:",
        accuracy
    )
    # ======================
    # 4. 保存预测结果
    # ======================
    y_true = []
    y_pred = []
    for images, labels in test_ds:
        predictions = model.predict(
            images,
            verbose=0
        )
        pred_classes = np.argmax(
            predictions,
            axis=1
        )
        y_true.extend(
            labels.numpy()
        )
        y_pred.extend(
            pred_classes
        )
    return (
        y_true,
        y_pred
    )
def plot_confusion_matrix(
        y_true,
        y_pred
):
    class_names = [
        "SUV",
        "bus",
        "family sedan",
        "fire engine",
        "heavy truck",
        "jeep",
        "minibus",
        "racing car",
        "taxi",
        "truck"
    ]
    cm = confusion_matrix(
        y_true,
        y_pred
    )
    plt.figure(
        figsize=(10,8)
    )
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "True"
    )
    plt.title(
        "Confusion Matrix"
    )
    save_path = os.path.join(
        RESULT_PATH,
        "confusion_matrix.png"
    )
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    print(
        "混淆矩阵保存:",
        save_path
    )
def plot_history():
    history_path = os.path.join(
        RESULT_PATH,
        "history.json"
    )
    with open(
        history_path,
        "r"
    ) as f:

        history = json.load(f)
    plt.figure()
    plt.plot(
        history["accuracy"],
        label="train accuracy"
    )
    plt.plot(
        history["val_accuracy"],
        label="val accuracy"
    )
    plt.legend()
    plt.title(
        "Accuracy"
    )
    plt.savefig(
        os.path.join(
            RESULT_PATH,
            "accuracy.png"
        )
    )
    plt.figure()
    plt.plot(
        history["loss"],
        label="train loss"
    )
    plt.plot(
        history["val_loss"],
        label="val loss"
    )
    plt.legend()
    plt.title(
        "Loss"
    )
    plt.savefig(
        os.path.join(
            RESULT_PATH,
            "loss.png"
        )
    )
def main():
    y_true, y_pred = evaluate_model()
    plot_confusion_matrix(
        y_true,
        y_pred
    )
    plot_history()

if __name__ == "__main__":

    main()