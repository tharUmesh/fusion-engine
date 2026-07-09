# HRI Scenarios & Motion Intent Mapping

This document provides a detailed explanation of the **39 HRI intent scenarios** across Classroom and Kitchen contexts and describes how the upgraded **8 human motion classes** fit into this multi-cue framework.

---

## 🧩 The 4-Cue HRI Intention Framework

User intention is rarely determined by a single action. In modern Human-Robot Interaction (HRI), predicting a user's intent is done by fusing **four distinct behavioral cues**:

$$\text{Context} + \text{Emotion} + \text{Gesture} + \text{Motion} \longrightarrow \text{User Intent}$$

1. **Context (Environment)**: Where is the interaction happening? (e.g., *Classroom* vs. *Kitchen*).
2. **Emotion (Affective State)**: How does the user feel? (e.g., *Happy*, *Sad*, *Angry*, *Surprised*, *Neutral*, *Fear*).
3. **Gesture (Fine-grained Action)**: What are the hands/arms doing? (e.g., *Raise hand*, *Wave*, *Thumbs up/down*, *Beckoning*, *Pointing*).
4. **Motion (Body Trajectory)**: How is the body moving in space relative to the robot? (e.g., *Walking*, *Walk Across*, *Sitting*, *Standing*, *Frozen*). **This module is responsible for detecting the Motion cue.**

---

## 🏫 Classroom Scenarios (1–20)

In the Classroom, the robot functions as a teaching assistant. The student's motions help the robot understand academic engagement, social interaction, and safety alerts.

### Key Scenarios & Motion Mapping:

*   **Scenario 1: Ask a Question / Request Help**
    *   *Cues*: Classroom + Neutral Emotion + Raise Hand (Gesture) + **Sitting Still (Motion)**
    *   *Intent*: Help Request (Ask a question)
    *   *Role of Motion*: Confirming the student is seated, quiet, and waiting to be addressed.
*   **Scenario 2: Student Greeting**
    *   *Cues*: Classroom + Happy Emotion + Wave (Gesture) + **Walking (Motion)**
    *   *Intent*: Social Interaction (Greeting)
    *   *Role of Motion*: Approach translation triggers the robot's greeting protocol.
*   **Scenario 4: Student Working**
    *   *Cues*: Classroom + Neutral Emotion + Writing (Gesture) + **Leaning Forward (Motion)**
    *   *Intent*: Academic engagement (Working on task)
    *   *Role of Motion*: Detecting the upper body leaning down/forward toward a desk.
*   **Scenario 6: Startled / Panic Retreat**
    *   *Cues*: Classroom + Fear/Surprise Emotion + Hands Up (Gesture) + **Walking (Motion)**
    *   *Intent*: Safety/Defensive (Retreating from danger)
    *   *Role of Motion*: Backward displacement triggers a safety shutdown or emergency alert on the robot.
*   **Scenario 8: Passing By**
    *   *Cues*: Classroom + Neutral Emotion + No Gesture + **Walk Across (Motion)**
    *   *Intent*: Passively transitioning (Ignoring robot)
    *   *Role of Motion*: Lateral walking velocity tells the robot not to interrupt the student.
*   **Scenario 13: Exhaustion / Disengagement**
    *   *Cues*: Classroom + Sad/Neutral Emotion + Head on desk + **Sitting Still (Motion)**
    *   *Intent*: Physical state check (Tiredness/Sickness)
    *   *Role of Motion*: Long duration of zero velocity with a seated posture.
*   **Scenario 19: Leaving Class (Active Exit)**
    *   *Cues*: Classroom + Neutral Emotion + Wave (Gesture) + **Walk Across/Walking (Motion)**
    *   *Intent*: Exit Classroom (Leaving)

---

## 🍳 Kitchen Scenarios (21–39)

In the Kitchen, the robot acts as a cooking assistant. Motions are crucial for safety (e.g., running around hot items) and cooperative meal preparation.

### Key Scenarios & Motion Mapping:

*   **Scenario 21: Cooking/Prepping**
    *   *Cues*: Kitchen + Neutral Emotion + Chopping/Stirring (Gesture) + **Standing Still (Motion)**
    *   *Intent*: Active task execution (Cooking)
    *   *Role of Motion*: Person is stationary at a counter; fine hand motions are active.
*   **Scenario 23: Hot Item Danger / Defensive Step**
    *   *Cues*: Kitchen + Surprised Emotion + Hands pulled back (Gesture) + **Walking (Motion)**
    *   *Intent*: Safety warning (Stepping back from heat/splatter)
    *   *Role of Motion*: Immediate backward displacement triggers the robot to offer oven mitts or shut off the stove.
*   **Scenario 30: Running Hazard**
    *   *Cues*: Kitchen + Playful/Anxious Emotion + No Gesture + **Run (Fast Movement) (Motion)**
    *   *Intent*: Safety Alert (Running in kitchen)
    *   *Role of Motion*: High-velocity trajectory triggers a vocal safety warning from the robot ("Please do not run in the kitchen").
*   **Scenario 34: Threat / Aggression Freeze**
    *   *Cues*: Kitchen + Fear Emotion + Rigid Posture (Gesture) + **Frozen/Rigid Stand (Motion)**
    *   *Intent*: Stress/Freeze Response
    *   *Role of Motion*: Extremely low coordinate variance (rigid freeze) compared to natural standing sway, indicating high tension.
*   **Scenario 38: Cooperating (Handing over tool)**
    *   *Cues*: Kitchen + Happy/Neutral Emotion + Extend Hand (Gesture) + **Walking (Motion)**
    *   *Intent*: Object Handover
    *   *Role of Motion*: Approach velocity coordinates tool exchange range.

---

## 📊 Motion Class Mapping Matrix

This matrix maps our **8 motion classes** to the target HRI Intent scenarios:

| Class ID | Motion Class | Typical Posture | Associated Intent Scenarios | Robot Action |
| :---: | :--- | :---: | :--- | :--- |
| **0** | **Sitting Still** | Sitting | Classroom Help (#1), Writing (#4), Sleeping (#13), Seated Chat (#14) | Address student, lower speaking volume |
| **1** | **Standing Still** | Standing | Kitchen Chopping (#21), Washing (#22), Waiting for tool (#35) | Standby, observe hands, hold tray |
| **2** | **Walking** | Standing | Greeting (#2), Approaching table (#16), recoil (#23), handover (#38) | Face user, initiate greeting/handover/alarm |
| **3** | **Walk Across** | Standing | Walking past (#8), Transitioning (#17), Crossing room (#25) | Maintain path, yield right of way |
| **4** | **Run Backward** | Standing | Sudden danger panic (#19) | Stop all motion, safety shutdown |
| **5** | **Run (Fast Movement)** | Standing | Run in kitchen hazard (#30) | Issue safety voice prompt ("Please walk") |
| **6** | **Leaning Forward** | Standing/Sitting | Working/Inspecting (#11), Leaning on counter (#28) | Present details closely, tilt robot screen |
| **7** | **Frozen/Rigid Stand** | Standing | Aggressive threat freeze (#34), Anxious hesitation (#36) | Back away slowly, adopt friendly posture |

---

## 🔄 Live Inference & Multi-Cue Integration

During live deployment, our module outputs predictions at **30 FPS**:

```
[Camera Frame] ──> [MediaPipe Pose] 
                       │
                       ├──> [Rule-based Posture Classifier] ──> Posture: "Standing"
                       │
                       └──> [Torso Translation Engine]      ──> Motion:  "Walking" (80%)
```

These values are written directly to the shared pipeline database or ROS topics:
*   `active_pose = "Standing"`
*   `active_motion = "Walking"`
*   `motion_confidence = 0.80`

A separate **Intent Fusion Model** (such as a multi-modal Transformer or Decision Tree) reads these motion values alongside the Context (e.g., Classroom), Emotion (e.g., Happy), and Gesture (e.g., Wave) to output the final Intention prediction: **Social Greeting (Say Hello)**.
