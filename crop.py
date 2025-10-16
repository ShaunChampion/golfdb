import sys
import os
from moviepy import VideoFileClip

def crop_mp4_to_160(input_mp4, output_mp4=None):
    if output_mp4 is None:
        base, _ = os.path.splitext(input_mp4)
        output_mp4 = base + "_cropped.mp4"

    with VideoFileClip(input_mp4) as clip:
        width, height = clip.size
        crop_size = min(width, height)
        x1 = (width - crop_size) // 2
        y1 = (height - crop_size) // 2
        
        # In moviepy 2.x, use crop method with different parameters
        cropped = clip.cropped(x1=x1, y1=y1, width=crop_size, height=crop_size)
        resized = cropped.resized((160, 160))
        resized.write_videofile(output_mp4, audio_codec='aac')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python crop.py <input.mp4> [output.mp4]")
        sys.exit(1)
    input_mp4 = sys.argv[1]
    output_mp4 = sys.argv[2] if len(sys.argv) > 2 else None
    crop_mp4_to_160(input_mp4, output_mp4)