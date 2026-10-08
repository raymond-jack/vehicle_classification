import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2

from utils.config import (
    IMG_SIZE,
    NUM_CLASSES,
    DROPOUT_RATE,
    BACKBONE,
)


def build_custom_cnn(num_classes=NUM_CLASSES, dropout=DROPOUT_RATE):

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")
    x = inputs
    filters = [32, 64, 128, 256, 512]
    for i, f in enumerate(filters):
        x = layers.Conv2D(
            f, (3, 3), padding="same",
            kernel_initializer="he_normal",
            name=f"conv{i + 1}",
        )(x)
        x = layers.Activation("relu", name=f"relu{i + 1}")(x)
        x = layers.MaxPooling2D((2, 2), name=f"pool{i + 1}")(x)
        # 高通道层后加 Dropout 抑制过拟合
        if f >= 128:
            x = layers.Dropout(0.25, name=f"drop{i + 1}")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu",
                     kernel_initializer="he_normal", name="dense_1")(x)
    x = layers.Dropout(dropout, name="drop_head")(x)
    outputs = layers.Dense(
        num_classes, activation="softmax", name="predictions"
    )(x)
    return Model(inputs, outputs, name="vehicle_custom_cnn")


def _build_backbone(name, input_tensor):

    scaled = layers.Lambda(lambda t: t * 2.0 - 1.0, name="scale_to_imagenet")(
        input_tensor
    )
    if name == "EfficientNetB0":
        base = EfficientNetB0(
            include_top=False, weights="imagenet",
            input_tensor=scaled, pooling=None,
        )
    elif name == "MobileNetV2":
        base = MobileNetV2(
            include_top=False, weights="imagenet",
            input_tensor=scaled, pooling=None,
        )
    else:
        raise ValueError(f"不支持的骨干网络: {name}")
    return base.output, base


def build_transfer_model(
    num_classes=NUM_CLASSES,
    backbone=BACKBONE,
    trainable_backbone=False,
    fine_tune_at=None,
):

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")
    backbone_out, base = _build_backbone(backbone, inputs)
    x = layers.GlobalAveragePooling2D(name="gap")(backbone_out)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(DROPOUT_RATE, name="drop_head_1")(x)
    x = layers.Dense(256, activation="relu", name="dense_256_2")(x)
    x = layers.Dropout(DROPOUT_RATE, name="drop_head_2")(x)
    outputs = layers.Dense(
        num_classes, activation="softmax", name="predictions"
    )(x)
    model = Model(inputs, outputs, name=f"vehicle_{backbone}")
    base.trainable = trainable_backbone
    if trainable_backbone and fine_tune_at is not None:
        total = len(base.layers)
        for layer in base.layers[: total - fine_tune_at]:
            layer.trainable = False
    return model


def build_model(num_classes=NUM_CLASSES, **kwargs):

    if BACKBONE == "CustomCNN":
        return build_custom_cnn(num_classes=num_classes)
    return build_transfer_model(num_classes=num_classes, backbone=BACKBONE, **kwargs)


def compile_model(model, learning_rate=1e-3, label_smoothing=0.0):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
def main():
    model = build_model()
    compile_model(model)
    model.summary()
if __name__ == "__main__":
    main()
