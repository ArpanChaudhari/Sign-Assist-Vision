import onnx

print("Loading the split model...")
model = onnx.load("model.onnx")

print("Merging and repackaging for the web...")
# This forces ONNX to pack all the data into a single, clean file
onnx.save_model(model, "model_web.onnx", save_as_external_data=False, all_tensors_to_one_file=True)

print("Created model_web.onnx successfully!")