import os
import json
import tensorflow as tf

from utils.config import (
    MODEL_PATH,
    RESULT_PATH,
    HEAD_EPOCHS,
    FINETUNE_EPOCHS,
    HEAD_LEARNING_RATE,
    FINETUNE_LEARNING_RATE,
    FINETUNE_AT,
    LABEL_SMOOTHING,
    BACKBONE,
)

import importlib

data_module = importlib.import_module("02_data_preprocessing")
get_prepared_data = data_module.get_prepared_data

model_module = importlib.import_module("03_model_build")
build_model = model_module.build_model
compile_model = model_module.compile_model
build_transfer_model = model_module.build_transfer_model


def make_callbacks():
    """早停 + 学习率衰减 + 保存最佳（按验证准确率）"""
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def train():
    # 1. 数据
    train_ds, val_ds, test_ds, class_names = get_prepared_data()
    print(f"\n类别数量: {len(class_names)}  -> {class_names}")

    # 2. 按骨干类型选择训练策略
    if BACKBONE == "CustomCNN":
        print("\n======================")
        print("训练策略：自研 CNN 从零训练（离线）")
        print("======================")
        model = build_model()
        compile_model(
            model,
            learning_rate=HEAD_LEARNING_RATE,
            label_smoothing=LABEL_SMOOTHING,
        )
        model.summary()
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=HEAD_EPOCHS + FINETUNE_EPOCHS,
            callbacks=make_callbacks(),
        )
    else:
        print("\n======================")
        print("阶段一：冻结骨干，训练分类头")
        print("======================")
        model = build_transfer_model(trainable_backbone=False)
        compile_model(
            model,
            learning_rate=HEAD_LEARNING_RATE,
            label_smoothing=LABEL_SMOOTHING,
        )
        model.summary()
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=HEAD_EPOCHS,
            callbacks=make_callbacks(),
        )

        print("\n======================")
        print(f"阶段二：解冻末尾 {FINETUNE_AT} 层骨干，低学习率微调")
        print("======================")
        model = build_transfer_model(
            trainable_backbone=True, fine_tune_at=FINETUNE_AT
        )
        compile_model(
            model,
            learning_rate=FINETUNE_LEARNING_RATE,
            label_smoothing=LABEL_SMOOTHING,
        )
        model.load_weights(MODEL_PATH)
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=FINETUNE_EPOCHS,
            callbacks=make_callbacks(),
        )

    # 3. 独立测试集评估
    print("\n======================")
    print("独立测试集评估")
    print("======================")
    loss, accuracy = model.evaluate(test_ds, verbose=0)
    print(f"Test Loss: {loss:.4f}   Test Accuracy: {accuracy:.4f}")

    # 4. 保存训练摘要
    summary = {
        "backbone": BACKBONE,
        "class_names": class_names,
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
    }
    with open(
        os.path.join(RESULT_PATH, "train_summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    print("\n======================")
    print("训练完成")
    print("最佳模型:", MODEL_PATH)
    print("测试集准确率: {:.2%}".format(accuracy))
    print("======================")


if __name__ == "__main__":
    train()
