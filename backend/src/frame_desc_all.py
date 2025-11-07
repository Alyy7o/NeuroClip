import argparse
import json
import subprocess
import os
from PIL import Image
import open_clip
import torch
import time
import numpy as np

# Initialize the model
start_model_time = time.perf_counter()
model, _, transform = open_clip.create_model_and_transforms(
    "coca_ViT-L-14",
    pretrained="mscoco_finetuned_laion2B-s13B-b90k"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
end_model_time = time.perf_counter()
print(f"Model loaded and moved to device={device} in {end_model_time - start_model_time:.1f}s")

def output_generate(image):
    im = transform(image).unsqueeze(0).to(device)
    im = im.float()
    # Use autocast only when CUDA is available
    if device.type == 'cuda':
        ctx = torch.cuda.amp.autocast()
    else:
        # no-op context manager for CPU
        from contextlib import nullcontext
        ctx = nullcontext()

    with torch.no_grad(), ctx:
        t0 = time.perf_counter()
        generated = model.generate(im, seq_len=20)
        t1 = time.perf_counter()
    # log generation time for diagnostics
    print(f"generated text in {t1 - t0:.2f}s")
    return open_clip.decode(generated[0].detach()).split("<end_of_text>")[0].replace("<start_of_text>", "")


def get_image(frames, video_file, folder_path):
    ffmpeg_commands = []
    for i, frame in enumerate(frames):
        ffmpeg_commands.append(f"ffmpeg -y -ss {frame} -i {video_file} -vframes 1 {folder_path}/{frame}.jpg")
    # run ffmpeg commands sequentially but time them; consider parallelizing if many timestamps
    for command in ffmpeg_commands:
        t0 = time.perf_counter()
        subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t1 = time.perf_counter()
        print(f"ffmpeg command finished in {t1 - t0:.2f}s")

def process_video_frames(video_file, json_file, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(json_file) as f:
        data = json.load(f)
    for sent in data['sentences']:
        sent_start = time.perf_counter()
        frames = []
        starttime = sent['starttime']
        endtime = sent['endtime']
        frames.append(starttime)
        frames.append(endtime)
        midtime = (float(starttime) + float(endtime)) / 2
        frames.append(midtime)
        if sent.get('verbs'):
            for verb in sent['verbs']:
                if verb.get('vstart') not in frames:
                    frames.append(verb['vstart'])

        # extract frames for this sentence
        get_image(frames, video_file, folder_path)

        frame_data = []
        # iterate only over files created for this sentence (simple filter by extension)
        for filename in [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]:
            file_path = os.path.join(folder_path, filename)

            if os.path.isfile(file_path):
                with Image.open(file_path) as image:
                    frame_text = output_generate(image)
                print('text', frame_text, file_path)
                frame_data.append(frame_text)
                os.remove(file_path)
        frame_data = np.unique(frame_data)
        sent['frame_data'] = frame_data.tolist()
        sent_end = time.perf_counter()
        print(f"processed sentence in {sent_end - sent_start:.2f}s")
    
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=1)
        
def main():

    video_files = [f for f in os.listdir("input_files") if f.endswith('.mp4')]
    
    for file_name in video_files:
        input_file = os.path.join("input_files", file_name)
        json_file = os.path.join("output_data", f"{file_name[:-4]}.v4.json")
        process_video_frames(input_file, json_file, "frames/")

if __name__ == "__main__":
    main()

