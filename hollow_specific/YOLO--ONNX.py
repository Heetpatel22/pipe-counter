from ultralytics import YOLO


model = YOLO("best_hollow.pt")
model.export(format="onnx", dynamic=True)
