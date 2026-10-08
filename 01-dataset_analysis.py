import os
from PIL import Image
import matplotlib.pyplot as plt
import random
from utils.config import DATA_PATH

def get_classes(data_path):
    """
    获取数据集中所有类别名称
    """
    classes = [
        folder
        for folder in os.listdir(data_path)
        if os.path.isdir(
            os.path.join(data_path, folder)
        )
    ]
    return classes
def count_images(data_path, classes):
    """
    统计每个类别图片数量
    """
    class_counts = {}
    image_extensions = (
        ".jpg",
        ".jpeg",
        ".png"
    )
    for cls in classes:
        class_path = os.path.join(
            data_path,
            cls
        )
        images = [
            img
            for img in os.listdir(class_path)
            if img.lower().endswith(
                image_extensions
            )
        ]


        class_counts[cls] = len(images)
    return class_counts
def print_dataset_info(classes, class_counts):
    print("\n========== 数据集分析 ==========")
    print(
        f"类别数量: {len(classes)}"
    )
    print("\n类别图片数量:")
    for cls,count in class_counts.items():
        print(
            f"{cls:<15}: {count:>5} 张"
        )
    total = sum(
        class_counts.values()
    )
    print(
        "\n总图片数量:",
        total
    )
    print(
        "================================"
    )
def analyze_image_size(data_path, classes):
    """
    分析图片尺寸
    """
    size_counts = {}
    image_extensions = (
        ".jpg",
        ".jpeg",
        ".png"
    )
    for cls in classes:
        class_path = os.path.join(
            data_path,
            cls
        )
        for img_name in os.listdir(class_path):

            if not img_name.lower().endswith(
                image_extensions
            ):
                continue
            img_path = os.path.join(
                class_path,
                img_name
            )
            try:
                with Image.open(img_path) as img:
                    size = img.size
            except Exception as e:
                print(f"图片读取失败：{img_path}")
                continue
            if size not in size_counts:
                size_counts[size] = 0
            size_counts[size] += 1
    return size_counts
def plot_class_ratio(class_counts):
    """
    绘制车辆类别比例饼图
    """
    labels = list(
        class_counts.keys()
    )
    values = list(
        class_counts.values()
    )
    plt.figure(
        figsize=(8, 8)
    )
    plt.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title(
        "Vehicle Dataset Class Ratio"
    )
    plt.show()
def main():
    print(
        "当前数据集路径:"
    )
    print(DATA_PATH)
    classes = get_classes(
        DATA_PATH
    )
    class_counts = count_images(
        DATA_PATH,
        classes
    )
    print_dataset_info(
        classes,
        class_counts
    )
    size_counts = analyze_image_size(
        DATA_PATH,
        classes
    )
    print("\n图片尺寸统计:")

    for size, count in size_counts.items():
        print(
            f"{size}: {count}张"
        )
    plot_class_ratio(
        class_counts
    )

if __name__ == "__main__":
    main()