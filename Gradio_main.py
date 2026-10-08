import os
import sys
import cv2
import numpy as np
from app.camera import Camera
import gradio as gr

# 把项目根目录加入路径，便于直接运行本文件
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.vehicle_predictor import VehiclePredictor
from utils.config import MODEL_PATH, TEST_PATH


def make_predictor():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"未找到模型文件: {MODEL_PATH}，请先运行 04_train.py 训练模型。"
        )
    return VehiclePredictor(MODEL_PATH)


predictor = make_predictor()
camera = Camera(0)
CAMERAS = {
    "电脑内置摄像头": 0,
    "外接摄像头": 1,
}


def _draw_label(rgb_img, name, conf):
    """在图片左上角绘制预测结果（RGB 图）"""
    canvas = rgb_img.copy()
    text = f"{name}  {conf:.1%}"
    cv2.putText(
        canvas, text,
        (12, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2, (0, 200, 0), 2, cv2.LINE_AA,
    )
    return canvas


def classify(img):
    """
    Gradio 回调：img 为 RGB numpy 数组（H,W,3）或 None。
    返回 (Top-K 字典用于 gr.Label, 带标注的 RGB 图)。
    """
    if img is None:
        return {}, None
    # Gradio 给的是 RGB，预测器内部按 BGR 处理
    frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    topk = predictor.predict_topk(frame_bgr, k=3)
    label_dict = {name: round(float(conf), 4) for name, conf in topk}
    name, conf = topk[0]
    out_img = _draw_label(img, name, conf)
    return label_dict, out_img


def change_camera(name):
    camera_id = CAMERAS[name]

    if camera.switch(camera_id):
        return f"{name}打开成功"

    return f"{name}打开失败"


last_result = {
    "topk": [],
    "label": "",
    "conf": 0
}
counter = 0


def camera_predict():
    global last_result
    frame = camera.read()
    if frame is None:
        return None, {}
    topk = predictor.predict_topk(
        frame,
        k=3
    )
    # 每次更新
    last_result["topk"] = topk
    last_result["label"] = topk[0][0]
    last_result["conf"] = topk[0][1]

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    rgb = _draw_label(
        rgb,
        last_result["label"],
        last_result["conf"]
    )

    result = {
        n: float(c)
        for n, c in last_result["topk"]
    }

    return rgb, result


# 示例图片（testimg 目录）
EXAMPLES = [
    os.path.join(TEST_PATH, f)
    for f in ("bus_test.jpg", "SUV_test.jpg", "truck_test.jpg")
    if os.path.exists(os.path.join(TEST_PATH, f))
]

with gr.Blocks(title="车辆分类识别系统") as demo:
    gr.Markdown("# 🚗 车辆图像分类识别系统")
    gr.Markdown(
        "上传一张车辆图片（或调用摄像头拍照），模型将返回 Top-3 预测类别与置信度。"
    )

    with gr.Tab("图片识别"):
        with gr.Row():
            img_in = gr.Image(
                label="车辆图片",
                type="numpy",
                sources=["upload", "clipboard"],
                height=420,
            )
        with gr.Row():
            label_out = gr.Label(num_top_classes=3, label="预测结果 (Top-3)")
            img_out = gr.Image(label="识别标注", type="numpy", height=420)
        if EXAMPLES:
            gr.Examples(
                examples=EXAMPLES,
                inputs=img_in,
                outputs=[label_out, img_out],
                fn=classify,
                cache_examples=False,
            )

    with gr.Tab("摄像头识别"):
        camera_select = gr.Dropdown(
            choices=list(CAMERAS.keys()),
            value="电脑内置摄像头",
            label="选择摄像头"
        )

        camera_status = gr.Textbox(
            label="状态"
        )

        cam_out = gr.Image(
            label="实时画面",
            type="numpy"
        )

        cam_label = gr.Label(
            num_top_classes=3,
            label="预测结果"
        )

        camera_select.change(
            change_camera,
            inputs=camera_select,
            outputs=camera_status
        )

        timer = gr.Timer(0.1)

        timer.tick(
            camera_predict,
            outputs=[
                cam_out,
                cam_label
            ]
        )
    # 绑定回调
    img_in.change(classify, inputs=img_in, outputs=[label_out, img_out])
    img_in.clear(lambda: ({}, None), None, [label_out, img_out])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
