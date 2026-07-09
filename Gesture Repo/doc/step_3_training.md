# Step 3: Neural Classifier Training

This step describes the training script, model architecture, and the rotation data augmentation technique.

---

## 🏗️ Neural Network Architecture (MLP)

The static hand pose classifier is a lightweight Multi-Layer Perceptron (MLP) built in Keras:

*   **Input Layer**: 42 units (21 landmarks $\times$ 2 coordinates)
*   **Dropout Layer (20%)**: Regularization to prevent overfitting
*   **Dense Hidden Layer**: 64 units, ReLU activation
*   **Dropout Layer (30%)**: Regularization
*   **Dense Hidden Layer**: 32 units, ReLU activation
*   **Dropout Layer (30%)**: Regularization
*   **Dense Hidden Layer**: 16 units, ReLU activation
*   **Dense Output Layer**: 6 units, Softmax activation (corresponds to the 6 target classes)

---

## 🔄 Rotation Data Augmentation

Since the HaGRID pointing dataset only features hands pointing vertically upwards, a model trained on raw data fails when the user points sideways or downwards.

To solve this, [train_keypoint.py](file:///d:/FYP/FYP_Motion%20&%20Gesture/Gesture_final/train/train_keypoint.py) applies **rotation augmentation** to the training coordinates:
*   **Class 0 (Open Palm), Class 1 (Close), Class 2 (Pointer)**: Augmented with 5 random rotations spanning the full **-180 to +180 degrees**. This makes pointing completely direction-invariant.
*   **Class 3 (Thumbs Up), Class 4 (Thumbs Down)**: Restricted to small rotations (**-30 to +30 degrees**) to prevent them from flipping upside down and swapping labels.
*   **Class 5 (Beckoning)**: Augmented with random rotations from **-45 to +45 degrees**.

Augmentation is performed in-place around the wrist origin $(0,0)$ using a 2D rotation matrix:
$$\begin{pmatrix} x' \\ y' \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix}$$

---

## 📦 Zero-Dependency Data Partitioning

Instead of requiring `scikit-learn` to partition the dataset, the training script uses pure **NumPy** matrix slicing to split the dataset (75% training, 25% validation), minimizing third-party library overhead.

---

## 🏃 Execution Command

Run the trainer:
```powershell
.\env\Scripts\python.exe train/train_keypoint.py
```
*   **Callbacks**: Utilizes early stopping (stops training if validation loss does not improve for 30 epochs).
*   **Saved Weights**: Exports Keras weights to `model/keypoint_classifier/keypoint_classifier.hdf5`.
