from ultralytics import YOLO

model = YOLO("path/of/file/ after prun + distill")
model.export(format="tensort", dynamic=True, simplify=True, opset=12, imgsz=(640, 640), int8 = True) ## format = tflite

