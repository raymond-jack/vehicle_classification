import cv2


class Camera:
    """
    摄像头管理类
    """
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
    def open(self):
        """
        打开摄像头
        """
        self.cap = cv2.VideoCapture(
            self.camera_id
        )
        if not self.cap.isOpened():
            print(
                f"摄像头 {self.camera_id} 打开失败"
            )
            self.cap = None
            return False
        print(
            f"摄像头 {self.camera_id} 打开成功"
        )
        return True
    def read(self):
        """
        读取一帧图片
        返回:
        frame
        """
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        """
        释放摄像头
        """
        if self.cap:
            self.cap.release()
            self.cap = None
    def switch(self, camera_id):
        """
        切换摄像头
        """
        self.release()
        self.camera_id = camera_id
        return self.open()

    def is_opened(self):
        """
        判断摄像头状态
        """
        return (
            self.cap is not None
            and self.cap.isOpened()
        )

if __name__ == "__main__":
    camera = Camera(
        camera_id=0
    )
    if camera.open():
        while True:
            frame = camera.read()
            if frame is None:
                break
            cv2.imshow(
                "Camera Test",
                frame
            )
            key = cv2.waitKey(1)
            if key == 27:
                break
    camera.release()
    cv2.destroyAllWindows()