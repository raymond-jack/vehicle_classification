import os
import glob
import json
import numpy as np
import tensorflow as tf
from utils.config import (
    DATA_PATH,
    IMG_SIZE,
    BATCH_SIZE,
    SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)

# 让线程可复现
AUTOTUNE = tf.data.AUTOTUNE


def load_raw_dataset(data_path):
    """
    扫描目录，按 70/15/15 确定性划分 (train/val/test)。
    """
    # 1) 扫描目录 -> (文件路径, 标签)；类别按字母序，与原 class_names 顺序一致
    class_names = sorted(
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d))
    )
    filepaths, labels = [], []
    for idx, name in enumerate(class_names):
        dir_path = os.path.join(data_path, name)
        for fp in glob.glob(os.path.join(dir_path, "*")):
            if os.path.isfile(fp):
                filepaths.append(fp)
                labels.append(idx)

    # 2) 固定种子确定性混洗（保证可复现）
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(filepaths))
    filepaths = [filepaths[i] for i in perm]
    labels = [labels[i] for i in perm]

    # 3) 索引切分 70/15/15
    n = len(filepaths)
    n_test = int(n * TEST_RATIO)
    n_trainval = n - n_test
    n_val = int(n_trainval * VAL_RATIO / (TRAIN_RATIO + VAL_RATIO))
    test_fp, test_lb = filepaths[:n_test], labels[:n_test]
    val_fp, val_lb = filepaths[n_test:n_test + n_val], labels[n_test:n_test + n_val]
    train_fp, train_lb = filepaths[n_test + n_val:], labels[n_test + n_val:]

    # 4) 加载函数：读文件 -> 解码 -> resize
    def _load(fp, lb):
        img = tf.io.read_file(fp)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        return img, lb

    def _make(fps, lbs):
        ds = tf.data.Dataset.from_tensor_slices((fps, lbs))
        ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
        return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    train_ds = _make(train_fp, train_lb)
    val_ds = _make(val_fp, val_lb)
    test_ds = _make(test_fp, test_lb)

    assert len(class_names) == 10, f"期望 10 类，实际 {len(class_names)} 类"
    return train_ds, val_ds, test_ds, class_names


def augment_fn(image):
    # 几何：仅水平翻转（车辆左右对称，可接受）
    image = tf.image.random_flip_left_right(image)

    # 光度：亮度 / 对比度 / 饱和度 / 色相 轻度扰动
    image = tf.image.random_brightness(image, max_delta=0.12)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.image.random_saturation(image, lower=0.85, upper=1.15)
    image = tf.image.random_hue(image, max_delta=0.05)

    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def preprocess_dataset(dataset, augment=False):
    normalization = tf.keras.layers.Rescaling(1.0 / 255.0)

    def _map(x, y):
        x = normalization(x)
        if augment:
            x = augment_fn(x)
        return x, y

    dataset = dataset.map(_map, num_parallel_calls=AUTOTUNE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def save_class_names(class_names):
    """保存类别名称（字母序，与模型输出索引一致）"""
    path = "utils/class_names.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=4)
    return path


def get_prepared_data():
    """
    对外统一接口：训练/验证/测试集统一先归一化到 [0,1]，
    仅训练集叠加数据增强。
    """
    train_ds, val_ds, test_ds, class_names = load_raw_dataset(DATA_PATH)
    train_ds = preprocess_dataset(train_ds, augment=True)
    val_ds = preprocess_dataset(val_ds)
    test_ds = preprocess_dataset(test_ds)
    save_class_names(class_names)
    return train_ds, val_ds, test_ds, class_names


def main():
    train_ds, val_ds, test_ds, class_names = get_prepared_data()
    print(f"训练集 batch 数: {len(train_ds)}")
    print(f"验证集 batch 数: {len(val_ds)}")
    print(f"测试集 batch 数: {len(test_ds)}")
    print("\n类别映射（模型输出索引顺序）:")
    for index, name in enumerate(class_names):
        print(f"  {index} : {name}")

    for images, _ in train_ds.take(1):
        print("\n训练集图片 shape:", images.shape)
        print("训练集像素范围:", float(images.numpy().min()), "~", float(images.numpy().max()))
    for images, _ in val_ds.take(1):
        print("验证集像素范围:", float(images.numpy().min()), "~", float(images.numpy().max()))


if __name__ == "__main__":
    main()
