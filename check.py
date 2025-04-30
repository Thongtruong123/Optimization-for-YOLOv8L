from ultralytics import YOLO
model = YOLO('F:/distillation/yolov8_distillation/prune.pt')
for name, module in model.model.named_modules():
    print(name, module)
