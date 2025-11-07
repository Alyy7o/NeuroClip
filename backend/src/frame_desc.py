import argparse
import json
import subprocess
import os
import os as _os
from PIL import Image
import easyocr
import torch
import time
import numpy as np

# Initialize OCR reader (CPU by default)
reader = easyocr.Reader(['en'], gpu=False)

def output_generate(image):
    # OCR over the frame; returns list of [bbox, text, confidence]
    result = reader.readtext(np.array(image))
    texts = [text.strip() for (_, text, conf) in result if isinstance(text, str) and conf >= 0.5]
    return " ".join(texts)


def get_image(frames, video_file, folder_path):
    ffmpeg_commands = []
    for i, frame in enumerate(frames):
        ffmpeg_commands.append(f"ffmpeg -y -ss {frame} -i {video_file} -vframes 1 {folder_path}/{frame}.jpg")
    for command in ffmpeg_commands:
        subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def process_video_frames(video_file, json_file, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(json_file) as f:
        data = json.load(f)
    # Optional: process every Nth sentence to speed up (default 1)
    stride = int(_os.getenv('SENTENCE_STRIDE', '1'))
    for idx, sent in enumerate(data['sentences']):
        if stride > 1 and (idx % stride) != 0:
            continue
        frames = []
        starttime = sent['starttime']
        endtime = sent['endtime']
        midtime = (float(starttime) + float(endtime)) / 2
        # Time efficiency: only capture the midpoint frame
        frames = [midtime]
        
        get_image(frames, video_file, folder_path)
        
        frame_data = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            if os.path.isfile(file_path):
                image = Image.open(file_path)
                frame_text = output_generate(image)
                print('text',frame_text,file_path)
                frame_data.append(frame_text)
                os.remove(file_path)

        frame_data = np.unique(frame_data)
        sent['frame_data'] = frame_data.tolist()
    
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=1)


def main(video_file, json_file, folder_path):

    # Process video frames
    process_video_frames(video_file, json_file, folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video frames and generate frame data")
    parser.add_argument("video_file", type=str, help="Input video file path")
    parser.add_argument("json_file", type=str, help="Input JSON file path")
    parser.add_argument("folder_path", type=str, help="Folder path to store frames")
    args = parser.parse_args()

    main(args.video_file, args.json_file, args.folder_path)

