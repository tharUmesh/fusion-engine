"""
Batch Video Processor — HRI Motion Recognizer
==============================================
Processes all videos in one or more folders through action_recognizer.py.
Phone/portrait videos (.MOV) are handled automatically — no separate script needed.
The recognizer auto-detects orientation and applies the correct layout.
Live preview window is shown by default for every video. Press Q to skip to next.

Usage:
    # Process testVideo/ (default, with live UI)
    python run_all_and_save.py

    # Process testVideo2/ (phone videos, with live UI)
    python run_all_and_save.py --folder testVideo2

    # Process both folders at once (with live UI)
    python run_all_and_save.py --folder testVideo testVideo2

    # Custom output folder
    python run_all_and_save.py --folder testVideo2 --output output\\phone_output

    # Silent/headless mode (no windows, just save)
    python run_all_and_save.py --headless
"""

import os
import subprocess
import sys
import argparse

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.MOV', '.MP4', '.AVI', '.MKV')
SCRIPT_NAME = "action_recognizer.py"


def get_videos(folder):
    """Return sorted list of video filenames in a folder."""
    if not os.path.exists(folder):
        print(f"[WARN] Folder not found: '{folder}'")
        return []
    files = [f for f in os.listdir(folder) if f.endswith(VIDEO_EXTENSIONS)]
    # Try numeric sort (for testVideo/1.mp4, 2.mp4 ...), fall back to alpha
    try:
        files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        files.sort()
    return files


def process_folder(folder, output_dir, headless=False):
    """Run action_recognizer.py on every video in folder, save to output_dir."""
    videos = get_videos(folder)
    if not videos:
        print(f"  No video files found in '{folder}'.")
        return 0, 0

    os.makedirs(output_dir, exist_ok=True)
    done, failed = 0, 0

    for video in videos:
        video_path = os.path.join(folder, video)
        name_only = os.path.splitext(video)[0]
        output_path = os.path.join(output_dir, f"output_{name_only}.avi")

        print(f"\n  >>> {video}")
        print(f"      -> {output_path}")

        cmd = [
            sys.executable, SCRIPT_NAME,
            "--video", video_path,
            "--save", output_path,
            # Live UI shown by default. --headless suppresses it.
        ]
        if headless:
            cmd.append("--no-show")

        try:
            subprocess.run(cmd, check=True)
            done += 1
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Failed: {e}")
            failed += 1
        except KeyboardInterrupt:
            print("\n  Interrupted by user.")
            break

    return done, failed


def main():
    parser = argparse.ArgumentParser(
        description="Batch process videos through HRI Motion Recognizer (handles phone & regular videos automatically)"
    )
    parser.add_argument(
        "--folder", nargs="+", default=["testVideo"],
        metavar="DIR",
        help="One or more video folders to process (default: testVideo). "
             "Example: --folder testVideo testVideo2"
    )
    parser.add_argument(
        "--output", default="output",
        metavar="DIR",
        help="Root output directory (default: output). "
             "Each folder gets its own sub-directory when multiple folders used."
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Suppress live preview windows — save output only, no display."
    )
    args = parser.parse_args()

    folders = args.folder
    root_output = args.output

    print("=" * 60)
    print("  HRI Batch Video Processor")
    print("  Phone videos auto-detected — no separate script needed")
    print(f"  Live UI : {'OFF (headless mode)' if args.headless else 'ON  (press Q to skip to next video)'}")
    print("=" * 60)
    print(f"\n  Folders  : {', '.join(folders)}")
    print(f"  Output   : {root_output}/")
    print()

    total_done, total_failed = 0, 0

    for folder in folders:
        # Sub-folder per source when processing multiple at once
        if len(folders) > 1:
            sub_name = os.path.basename(os.path.normpath(folder))
            output_dir = os.path.join(root_output, sub_name)
        else:
            output_dir = root_output

        videos = get_videos(folder)
        print(f"[{folder}]  {len(videos)} video(s) found")
        done, failed = process_folder(folder, output_dir, headless=args.headless)
        total_done += done
        total_failed += failed

    print("\n" + "=" * 60)
    print(f"  Done: {total_done} video(s)  |  Failed: {total_failed}")
    print(f"  Outputs saved to: {root_output}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
