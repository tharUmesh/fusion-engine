#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HaGRID Dataset Extractor for Keypoint Classifier Training
===========================================================
Processes images from HaGRID sample dataset through MediaPipe Hands
to extract 21 hand landmarks, normalizes them, and saves to keypoint.csv.

Usage:
    python extract_dataset.py
    python extract_dataset.py --max_per_class 2000
    python extract_dataset.py --dataset_path /path/to/hagrid

CSV Format: label, x0, y0, x1, y1, ... x20, y20  (42 values + 1 label = 43 columns)
Labels: 0=Open, 1=Close, 2=Pointer, 3=OK
"""

import os
import cv2
import csv
import copy
import argparse
import itertools
import mediapipe as mp


def get_args():
    parser = argparse.ArgumentParser(description='Extract HaGRID dataset to keypoint.csv')
    parser.add_argument('--dataset_path', type=str,
                        default='D:/FYP/FYP_Tranformer/ourModelsprojects/gesture/gesture_detection/dataset/extracted/hagrid-sample-30k-384p/hagrid_30k',
                        help='Path to HaGRID image folders')
    parser.add_argument('--csv_path', type=str,
                        default='model/keypoint_classifier/keypoint.csv',
                        help='Output CSV file path')
    parser.add_argument('--max_per_class', type=int, default=1500,
                        help='Maximum samples per gesture class')
    parser.add_argument('--min_detection_confidence', type=float, default=0.5,
                        help='MediaPipe minimum detection confidence')
    return parser.parse_args()


def pre_process_landmark(landmark_list):
    """
    Normalize 21 hand landmarks to relative coordinates:
    1. Translate so wrist (landmark 0) is at origin
    2. Flatten to 1D list [x0, y0, x1, y1, ..., x20, y20]
    3. Normalize by max absolute value so all values in [-1, 1]
    """
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Step 1: Convert to relative coordinates (wrist = origin)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for index, _ in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y

    # Step 2: Flatten to 1D
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Step 3: Normalize by max absolute value
    max_value = max(list(map(abs, temp_landmark_list)))
    if max_value == 0:
        return temp_landmark_list  # Avoid division by zero
    temp_landmark_list = [n / max_value for n in temp_landmark_list]

    return temp_landmark_list


def main():
    args = get_args()

    # === HaGRID Folder → Label Mapping ===
    # These folders from HaGRID map to our 6 static gesture classes
    mapping = {
        'train_val_palm': 0,      # Open Palm
        'train_val_stop': 0,      # Open Palm (stop sign = open palm)
        'train_val_fist': 1,      # Close (fist)
        'train_val_one': 2,       # Pointer (index finger)
        'train_val_like': 3,      # Thumbs Up
        'train_val_dislike': 4,   # Thumbs Down
        'train_val_call': 5,      # Beckoning/Call
    }

    print("=" * 60)
    print("HaGRID -> Keypoint CSV Extractor")
    print("=" * 60)
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output CSV:   {args.csv_path}")
    print(f"Max per class: {args.max_per_class}")
    print(f"Mapping: {mapping}")
    print("=" * 60)

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
    )

    total_samples = 0

    # Open CSV for writing
    with open(args.csv_path, 'w', newline='') as f:
        writer = csv.writer(f)

        for folder_name, label in mapping.items():
            full_folder_path = os.path.join(args.dataset_path, folder_name)
            if not os.path.exists(full_folder_path):
                print(f"Warning: {full_folder_path} not found. Skipping.")
                continue

            print(f"\nProcessing '{folder_name}' -> Label {label}...")
            images = sorted([f for f in os.listdir(full_folder_path)
                           if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

            count = 0
            skipped = 0
            for image_name in images:
                if count >= args.max_per_class:
                    break

                image_path = os.path.join(full_folder_path, image_name)
                image = cv2.imread(image_path)
                if image is None:
                    skipped += 1
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Extract landmark coordinates (normalized 0-1 by MediaPipe)
                        landmark_list = []
                        for landmark in hand_landmarks.landmark:
                            landmark_list.append([landmark.x, landmark.y])

                        # Apply relative normalization
                        processed_landmarks = pre_process_landmark(landmark_list)

                        # Write: [label, x0, y0, x1, y1, ..., x20, y20]
                        writer.writerow([label, *processed_landmarks])
                        count += 1

                        if count >= args.max_per_class:
                            break
                else:
                    skipped += 1

                # Progress indicator
                if count % 100 == 0 and count > 0:
                    print(f"   {count}/{args.max_per_class} samples extracted...")

            total_samples += count
            print(f"   Done! {count} samples extracted, {skipped} images skipped.")

    hands.close()
    print(f"\n{'=' * 60}")
    print(f"Extraction complete!")
    print(f"   Total samples: {total_samples}")
    print(f"   Saved to: {args.csv_path}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
