# 
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Define the initial types for the ONNX model
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]

# Convert the scikit-learn model to ONNX
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Save the ONNX model to a file
with open("linear_regression.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())