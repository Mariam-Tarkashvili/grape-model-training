from tf2onnx.convert import from_saved_model

model_path = r"C:\Users\User\PycharmProjects\grape_backup\saved_model"
onnx_model, _ = from_saved_model(model_path, opset=13)

with open("../grape_disease/grape_leaf_spot_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("✅ Converted SavedModel to ONNX successfully")
