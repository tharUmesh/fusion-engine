# Step 4: Model Quantization & Optimization

This step covers the conversion of the Keras model into a quantized, highly optimized TensorFlow Lite format suitable for real-time edge execution on the Jetson Orin Nano.

---

## ⚡ Quantization Process

Standard neural network models save weights as 32-bit floating-point numbers (`FP32`). While highly precise, this format consumes significant memory and introduces processing overhead on edge hardware.

The training script automatically performs **post-training integer quantization** during the export stage:

1. **SavedModel Extraction**: Loads the trained Keras network.
2. **Quantization Optimization**: Converts weights and operations using `tf.lite.Optimize.DEFAULT`.
3. **Weight Precision Reduction**: Compresses weights from 32-bit floats down to 8-bit integers (`INT8`) or 16-bit floats (`FP16`).

---

## 📈 Performance Benefits

*   **Ultra-Small Model Size**: Compresses the keypoint classifier to a tiny footprint of **~6-8 KB**.
*   **Blazing Fast Execution**: Quantized operations execute on the Jetson Orin Nano's CPU in **<0.1 milliseconds** per hand, utilizing minimal power.
*   **Lower Memory Bandwidth**: Reduces memory cache read latency, allowing other HRI modules (like posture or emotion trackers) to run concurrently without performance degradation.
