# Step 5: Real-time Inference & Landmark Smoothing

This step explains how the application runs live classification and filters webcam noise to prevent coordinate jittering.

---

## 📈 Landmark Smoothing Filter (EMA)

Webcam image streams suffer from subtle pixel noise and compression artifacts. This causes MediaPipe's skeletal joint landmarks to vibrate slightly in place, resulting in jittery drawings ("dancing points") and unstable classifications.

To solve this, we implement an **Exponential Moving Average (EMA)** smoothing filter:

1. **State Association**: The filter tracks coordinate histories independently for each hand using the unique spatial slots (ID `0` and `1`) assigned by the spatial distance tracker.
2. **Mathematical Formula**: For each coordinate pair $P_t$, we compute the smoothed coordinates $S_t$ using a smoothing factor $\alpha$:
   $$S_t = \alpha P_t + (1 - \alpha) S_{t-1}$$
   - **Value of $\alpha$**: Set to `0.45`, which provides the optimal balance between completely removing coordinate jitter and avoiding lag.
3. **In-place Protobuf Replacement**: The smoothed coordinates are written back directly into the MediaPipe hand landmarks structures. This ensures that:
   - The green skeleton lines drawn on the video window are perfectly steady.
   - The input fed to the TFLite model is noise-free, preventing classification flickering.

---

## 🧬 Dynamic Intent Resolving

Static poses predicted by the model are integrated with scale-normalized temporal motion history to recognize active gestures:

*   **Scale-Invariant Waving**: The horizontal coordinates of the hand are tracked continuously. The movement range ($x_{range}$) is evaluated dynamically relative to the hand's width ($x_{range} > 0.30 \times hand\_width$). If the dominant hand is in the `Open Palm` (Class 0) state and changes direction at least 2 times, it triggers **Wave**.
*   **Scale-Invariant Beckoning**: Tracks the index finger tip projection height ($INDEX\_TIP.y - INDEX\_PIP.y$) normalized by the hand's scale (Wrist-to-Middle MCP distance). Excursion is checked scale-invariantly ($excursion > 0.15$). Hand motion coordinates are continuously tracked for any shape in `[0, 1, 2, 5, -1]` (covering curls and unknown frames) to detect a **Beckoning** motion.
*   **Flex-Pose Hand Raising & Arms Up**: Hand-raising states ("One Hand Raised" and "Arms Up") support **Open Palm (0)**, **Fist (1)**, and **Unknown/Other (-1)** hand shapes. A rolling vertical history ($y\_hist$, size 25) monitors the hand's vertical coordinates. It classifies the hand as raised if it is either held very high statically ($y < 0.35$) or transitioned from a lower start position to a higher end position ($delta\_y > 0.12$ and $current\_y < 0.45$).
*   **Thumbs Up / Thumbs Down**: Directly mapped from the TFLite classifier predictions (Classes 3 and 4) to confirm or reject commands.

---

## 📷 Frame Preprocessing (Rotation & Sizing)

Before passing the frame to MediaPipe and drawing utilities, both test and real-time scripts perform two preprocessing steps:
1. **EXIF Metadata Auto-Rotation**: Queries the video's rotation metadata property (index 48 or `cv.CAP_PROP_ORIENTATION_META`). If it is rotated (e.g. 90°, 180°, or 270° from phone recording), it applies the corresponding clockwise or counter-clockwise `cv.rotate()` call.
2. **Aspect Ratio Preserving Resizing**: Scales high-resolution frames (e.g., 4K phone recordings) down to a maximum dimension of **960 pixels** using area interpolation (`cv.INTER_AREA`). This ensures display windows fit cleanly on standard monitors and boosts MediaPipe and classification frame rates (FPS) drastically.
