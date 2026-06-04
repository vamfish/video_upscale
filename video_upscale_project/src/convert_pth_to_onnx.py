import torch
from spandrel import ModelLoader
wrapper = ModelLoader().load_from_file("RealESRGAN_x4plus.pth")
model = wrapper.model
model.eval()
dummy = torch.randn(1, 3, 64, 64)
torch.onnx.export(model, dummy, "RealESRGAN_x4plus.onnx",
                  opset_version=14,
                  input_names=["input"],
                  output_names=["output"],
                  dynamic_axes={"input": {0:"batch",2:"height",3:"width"},
                                 "output":{0:"batch",2:"height",3:"width"}})
print("✅ 转换成功: RealESRGAN_x4plus.onnx")
