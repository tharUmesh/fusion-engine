# Step 2: Keypoint Feature Extraction

This step details how raw image folders are processed to extract landmark coordinates and generate the training CSV file.

---

## ⚙️ Extraction Logic (`extract_dataset.py`)

Raw pixel images are highly sensitive to background noise and lighting conditions. To achieve domain invariance, we extract and normalize skeletal landmarks using **MediaPipe Hands**:

1. **Joint Landmarks Detection**: MediaPipe processes each frame to find 21 key points of the hand.
2. **Translation (Origin Normalization)**: The coordinates of all landmarks ($x_i, y_i$) are translated so that the **wrist (Landmark 0)** is at the origin $(0, 0)$:
   $$x_i' = x_i - x_0, \quad y_i' = y_i - y_0$$
3. **Flattening**: The translated coordinate points are flattened into a 1D array of 42 values:
   $$[x_0', y_0', x_1', y_1', \dots, x_{20}', y_{20}']$$
4. **Scale Normalization**: To ensure distance-invariance (so hands close to the camera have the same representation as hands far away), we divide all coordinates by the maximum absolute coordinate value in the vector, scaling them into the range $[-1, 1]$:
   $$\text{vector}_{\text{norm}} = \frac{\text{vector}}{\max(|x_j'|, |y_j'|)}$$

---

## 📄 Output Dataset Format (`keypoint.csv`)

The extracted landmarks are saved to `model/keypoint_classifier/keypoint.csv`.
*   **Format**: `label, x0, y0, x1, y1, ..., x20, y20` (43 columns: 1 label index + 42 normalized coordinates).
*   **Dataset Size**: 1,000 samples per mapping folder (7,000 samples total).

---

## 🏃 Execution Command

Run the extractor from the root of the project:
```powershell
.\env\Scripts\python.exe extract_dataset.py --max_per_class 1000
```
