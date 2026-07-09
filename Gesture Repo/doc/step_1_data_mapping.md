# Step 1: Data Preparation & Mappings

This step explains how raw gesture image categories map to the Human-Robot Interaction (HRI) target intent classes.

---

## 📂 Dataset Source: HaGRID 30k
The pipeline trains on a balanced subset of the **HaGRID (HAnd Gesture Recognition Image Dataset)**. The dataset is organized into folders, each containing raw JPEG/PNG images of specific hand gestures.

We utilize the local dataset folder located at:
`D:\FYP\FYP_Tranformer\ourModelsprojects\gesture\gesture_detection\dataset\extracted\hagrid-sample-30k-384p\hagrid_30k`

---

## 🗺️ Gesture Intent Mappings
To satisfy the HRI Scenarios (from the [HRI_Dataset_Table](file:///d:/FYP/FYP_Motion%20&%20Gesture/Gesture_final/HRI_Dataset_Table..pdf)), we map 7 specific HaGRID folders into **6 static hand pose classes** representing distinct states:

| Class ID | Pose Label | Source HaGRID Folders | Description / Intent Target |
| :--- | :--- | :--- | :--- |
| **0** | **Open Palm** | `train_val_palm`<br>`train_val_stop` | Hand open with extended fingers. Used for: `raise hand`, waving movements, and `both hands up` (alerts). |
| **1** | **Close (Fist)** | `train_val_fist` | Clenched fist shape. Used to denote neutral task states, resting hand, or a closed hand pose. |
| **2** | **Pointer** | `train_val_one` | Index finger extended vertically/horizontally, others curled. Used for pointing at items (spilled food, shelves, objects). |
| **3** | **Thumbs Up** | `train_val_like` | Thumb pointing up, fingers closed. Denotes confirmation, understanding, and positive feedback. |
| **4** | **Thumbs Down**| `train_val_dislike` | Thumb pointing down, fingers closed. Denotes help requests, task errors, disgust, or break requests. |
| **5** | **Beckoning** | `train_val_call` | Call/beckoning gesture shape. Used to identify requests for the robot to approach. |

---

## 🛠️ Verification
Verify that the HaGRID folders named above exist at the target path before running the extraction step.
