import os
import cv2
from base_camera import BaseCamera
from PicoDet import PicoDet
net = PicoDet(model_pb_path='coco/picodet_s_320_coco.onnx', label_path='coco/coco.names', prob_threshold=0.4, iou_threshold=0.3)


class Camera(BaseCamera):
    video_source = 0

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, img = camera.read()

            # process
            img = net.detect(img)
            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()
