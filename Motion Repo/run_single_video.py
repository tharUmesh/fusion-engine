"""
Interactive Single Video Runner — HRI Motion Recognizer
=======================================================
Lists all videos from testVideo/ and testVideo2/ in one menu.
Phone/portrait videos are handled automatically — just pick and run.

Usage:
    python run_single_video.py
"""

import os
import subprocess
import sys

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.mov', '.mp4', '.avi', '.mkv')
SCRIPT_NAME = "action_recognizer.py"

# All source folders — add more here if needed
VIDEO_FOLDERS = ["testVideo", "testVideo2"]


def collect_all_videos():
    """Gather videos from all folders into a flat list with source labels."""
    entries = []  # list of (display_label, filepath)
    for folder in VIDEO_FOLDERS:
        if not os.path.exists(folder):
            continue
        files = [f for f in os.listdir(folder) if f.lower().endswith(VIDEO_EXTENSIONS)]
        try:
            files.sort(key=lambda x: int(os.path.splitext(x)[0]))
        except ValueError:
            files.sort()
        for f in files:
            # Label shows source folder so user knows which is phone vs regular
            label = f"[{folder}]  {f}"
            entries.append((label, os.path.join(folder, f)))
    return entries


def select_and_run():
    print("=" * 60)
    print("  HRI Motion Recognizer — Select a Video")
    print("  Phone videos (.MOV) handled automatically")
    print("=" * 60)
    print(f"  Python: {sys.executable}\n")

    entries = collect_all_videos()

    if not entries:
        print("No videos found in any of:", VIDEO_FOLDERS)
        return

    # Display menu
    print("Available videos:\n")
    for i, (label, _) in enumerate(entries):
        print(f"  [{i+1:>3}]  {label}")
    print()

    # Selection
    try:
        val = input("Enter number or full filename/path: ").strip()
        if not val:
            print("No input provided.")
            return

        target_video = ""

        if val.isdigit():
            idx = int(val) - 1
            if 0 <= idx < len(entries):
                target_video = entries[idx][1]
            else:
                print(f"Invalid number. Enter 1–{len(entries)}.")
                return
        else:
            # Try as a path directly
            if os.path.isfile(val):
                target_video = val
            else:
                # Search all folders
                for folder in VIDEO_FOLDERS:
                    candidate = os.path.join(folder, val)
                    if os.path.isfile(candidate):
                        target_video = candidate
                        break
                if not target_video:
                    print(f"Could not find: {val}")
                    return

    except (ValueError, KeyboardInterrupt):
        print("\nCancelled.")
        return

    # Ask to save?
    print(f"\n>>> Running on: {target_video}")
    save_val = input("Save output? Enter filename (e.g. output/result.avi) or press Enter to skip: ").strip()

    cmd = [sys.executable, SCRIPT_NAME, "--video", target_video]
    if save_val:
        os.makedirs(os.path.dirname(save_val) if os.path.dirname(save_val) else ".", exist_ok=True)
        cmd += ["--save", save_val]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    select_and_run()
