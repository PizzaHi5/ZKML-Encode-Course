import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Generate some dummy data
X = np.random.rand(100, 1) * 10  # 100 samples, 1 feature
y = 2 * X + 1 + np.random.randn(100, 1) * 2  # y = 2x + 1 + noise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Define the initial types for the ONNX model
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]

# Convert the scikit-learn model to ONNX
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Save the ONNX model to a file
with open("linear_regression.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())