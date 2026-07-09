import csv
import numpy as np
import tensorflow as tf

dataset = 'model/keypoint_classifier/keypoint.csv'
model_save_path = 'model/keypoint_classifier/keypoint_classifier.hdf5'
tflite_save_path = 'model/keypoint_classifier/keypoint_classifier.tflite'
NUM_CLASSES = 6

print("Loading dataset...")
X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))

# Split train/test using numpy
indices = np.arange(X_dataset.shape[0])
np.random.seed(42)
np.random.shuffle(indices)

X_dataset = X_dataset[indices]
y_dataset = y_dataset[indices]

split_idx = int(0.75 * X_dataset.shape[0])
X_train, X_test = X_dataset[:split_idx], X_dataset[split_idx:]
y_train, y_test = y_dataset[:split_idx], y_dataset[split_idx:]

print(f"Original train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Rotation Augmentation
print("Applying rotation augmentation to handle pointing in any direction...")
def rotate_keypoints(landmarks, angle_deg):
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, -s], [s, c]])
    pts = landmarks.reshape(21, 2)
    rotated = np.dot(pts, R.T)
    return rotated.flatten()

X_aug = []
y_aug = []

for x_val, y_val in zip(X_train, y_train):
    # Keep original
    X_aug.append(x_val)
    y_aug.append(y_val)
    
    # Generate 5 augmented versions per sample
    if y_val in [0, 1, 2]:  # Open Palm, Close, Pointer (full 360 rotation)
        angles = np.random.uniform(-180, 180, size=5)
    elif y_val in [3, 4]:   # Thumbs Up, Thumbs Down (restricted to avoid label swapping)
        angles = np.random.uniform(-30, 30, size=5)
    else:                   # Beckoning
        angles = np.random.uniform(-45, 45, size=5)
        
    for angle in angles:
        X_aug.append(rotate_keypoints(x_val, angle))
        y_aug.append(y_val)

X_train = np.array(X_aug, dtype='float32')
y_train = np.array(y_aug, dtype='int32')

print(f"Augmented train shape: {X_train.shape}")
print(f"Target number of classes: {NUM_CLASSES}")

# Build sequential model
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(21 * 2, )),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.summary()

# Callbacks
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    model_save_path, verbose=1, save_weights_only=False)
es_callback = tf.keras.callbacks.EarlyStopping(patience=30, verbose=1)

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
print("Starting training...")
model.fit(
    X_train,
    y_train,
    epochs=500,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[cp_callback, es_callback]
)

# Evaluate
val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)
print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")

# Save
model.save(model_save_path, include_optimizer=False)

# Convert to TFLite
print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

with open(tflite_save_path, 'wb') as f:
    f.write(tflite_quantized_model)
print("TFLite model saved to:", tflite_save_path)
