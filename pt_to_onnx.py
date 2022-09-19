import onnx
from onnx import shape_inference
model = './runs/train/exp4/weights/best.onnx'
model_= './best_.onnx'
onnx.save(onnx.shape_inference.infer_shapes(onnx.load(model)), model_)