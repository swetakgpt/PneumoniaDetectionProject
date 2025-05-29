# convert_to_onnx.py
import tensorflow as tf
import tf2onnx
import onnx
import pickle
import numpy as np # For creating dummy input

# 1. Load your Keras model from model.pkl
MODEL_PKL_PATH = 'model.pkl'
ONNX_MODEL_PATH = 'pneumonia_model.onnx' # Output path for the ONNX model

print(f"Loading Keras model from: {MODEL_PKL_PATH}")
try:
    with open(MODEL_PKL_PATH, 'rb') as f:
        keras_model = pickle.load(f)
    print(f"Successfully loaded Keras model. Type: {type(keras_model)}")
except Exception as e:
    print(f"Error loading Keras model from pickle: {e}")
    exit()

# Verify it's a Keras Functional model (optional, but good check)
if not isinstance(keras_model, tf.keras.Model):
    print(f"Loaded model is not a tf.keras.Model instance. Type: {type(keras_model)}")
    # If it's a custom object wrapping a Keras model, you might need to extract the Keras model itself
    # e.g., if keras_model = loaded_object.model
    # exit() # Or handle as appropriate

# 2. Determine the input signature/specification for the model
# This is crucial for the tf2onnx converter.
# You need to know the expected shape and dtype of your model's input.
# For an image model, it's typically (batch_size, height, width, channels)
# Let's assume your pneumonia model expects 224x224 RGB images (like in your preprocess_image)
# For conversion, batch_size can often be dynamic (None) or fixed (e.g., 1).
# Using a dynamic batch size is generally preferred for flexibility.

# Example input spec for a model expecting (None, 224, 224, 3) float32 images:
input_signature = [tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name="input_image")]
# If your model has a different input name, find it via:
# print(keras_model.inputs) # This will show the Keras input tensors
# And adjust the 'name' in TensorSpec accordingly, or often it can be omitted if there's one input.

print(f"Keras model summary:")
keras_model.summary() # Print model summary to verify input/output shapes

print(f"Using input signature for ONNX conversion: {input_signature}")

# 3. Convert the Keras model to ONNX
# `opset` is important. 13 or 15 are common good choices. Higher opsets support more operations.
# If you encounter issues, you might try different opset versions.
try:
    print("Starting ONNX conversion...")
    # Use from_keras for tf.keras.Model instances
    model_proto, external_tensor_storage = tf2onnx.convert.from_keras(
        keras_model,
        input_signature=input_signature,
        opset=13, # Try 13, 15, or check tf2onnx docs for latest recommendations
        output_path=ONNX_MODEL_PATH
    )
    print(f"Model successfully converted to ONNX and saved at: {ONNX_MODEL_PATH}")
except Exception as e:
    print(f"Error during ONNX conversion: {e}")
    exit()

# 4. (Optional but Recommended) Verify the ONNX model
try:
    print("Verifying the ONNX model...")
    onnx_model = onnx.load(ONNX_MODEL_PATH)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid.")

    # 5. (Optional but Recommended) Test inference with ONNX Runtime
    print("Testing inference with ONNX Runtime...")
    import onnxruntime as ort
    import cv2 # For a quick image load and preprocess test

    # Prepare a dummy input or a sample image like your Flask app would
    # This needs to match the preprocessing your Flask app will do
    def preprocess_for_onnx_test(image_path, target_size=(224, 224)):
        img = cv2.imread(image_path)
        if img is None: return None
        img = cv2.resize(img, target_size)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0) # Add batch dimension: (1, 224, 224, 3)
        return img

    # Create or point to a sample image for testing
    # e.g., dummy_image_data = np.random.rand(1, 224, 224, 3).astype(np.float32)
    # Or use a real image path:
    # test_image_path = 'path/to/a/sample_test_image.jpg'
    # dummy_image_data = preprocess_for_onnx_test(test_image_path)

    # Using random data for simplicity here:
    dummy_image_data = np.random.rand(1, 224, 224, 3).astype(np.float32)


    if dummy_image_data is not None:
        ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
        
        # Get input and output names from the ONNX model
        input_name = ort_session.get_inputs()[0].name
        print(f"ONNX Model Input Name: {input_name}, Shape: {ort_session.get_inputs()[0].shape}")
        
        output_names = [output.name for output in ort_session.get_outputs()]
        print(f"ONNX Model Output Names: {output_names}, Shapes: {[output.shape for output in ort_session.get_outputs()]}")

        input_feed = {input_name: dummy_image_data}
        
        print(f"Running inference on dummy data with shape: {dummy_image_data.shape}")
        onnx_outputs = ort_session.run(output_names, input_feed)
        
        print(f"ONNX Runtime inference output (first output, first batch): {onnx_outputs[0][0]}")
        print("Inference test with ONNX Runtime completed.")
    else:
        print("Skipping ONNX Runtime inference test as dummy image data could not be prepared.")

except Exception as e:
    print(f"Error during ONNX model verification or ONNX Runtime test: {e}")

print("\nConversion script finished.")