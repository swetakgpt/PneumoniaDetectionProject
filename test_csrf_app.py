import keras2onnx
import onnx
from tensorflow.keras.models import load_model

# Load your model
model = load_model("CNN_Classification_1.h5")

# Convert to ONNX
onnx_model = keras2onnx.convert_keras(model, model.name)
onnx.save_model(onnx_model, "cnn_model.onnx")