import os
from ultralytics import YOLO
import torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():
    # model = YOLO(r'ultralytics/cfg/models/v8/yolov8s.yaml').load('runs/detect/yolov8s/weights/best.pt')
    model_s = YOLO("prune.pt")
    model_s.train(data="coco128.yaml", Distillation = None, loss_type='None', amp=False, imgsz=640, epochs=1, batch=4)


if __name__ == '__main__':
    main()
