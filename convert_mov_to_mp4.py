import os
import sys
import glob
import subprocess

def convert_mov_to_mp4(input_path, output_path=None):
    if os.path.isfile(input_path):
        mov_files = [input_path]
    else:
        mov_files = glob.glob(os.path.join(input_path, "*.mov"))

    if not mov_files:
        print("No .mov files found in", input_path)
        return

    for idx, mov_file in enumerate(mov_files):
        if output_path and os.path.isfile(input_path):
            mp4_file = output_path
        else:
            base = os.path.splitext(os.path.basename(mov_file))[0]
            out_dir = output_path if output_path and os.path.isdir(output_path) else os.path.dirname(mov_file)
            mp4_file = os.path.join(out_dir, base + ".mp4")
        print(f"Converting {mov_file} -> {mp4_file}")
        cmd = [
            "ffmpeg",
            "-i", mov_file,
            "-vcodec", "libx264",
            "-acodec", "aac",
            "-strict", "experimental",
            mp4_file
        ]
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_mov_to_mp4.py <input_folder_or_file> [output_folder_or_file]")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    convert_mov_to_mp4(input_path, output_path)