import os
from ultralytics import YOLO
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():
    model_t = YOLO('yolov8s.pt')  # the teacher model
    model_s = YOLO('F:/distillation/yolov8_distillation/prune.pt')  # the student model
    """
    Attributes:
        Distillation: the distillation model
        loss_type: mgd, cwd
        amp: Automatic Mixed Precision
    """

    

    model_s.train(data="coco128.yaml", Distillation=model_t.model, loss_type='mgd', amp=False, imgsz=640, epochs=5,
                  batch=1, device='cpu',  lr0=0.001)


if __name__ == '__main__':
    main()
