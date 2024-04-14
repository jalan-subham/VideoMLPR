import glob 
import os 
import tqdm

def preprocess_bol():
    with open("BagOfLies/vid_paths.txt", "r") as f:
        bol_vids_loc = [line.strip() for line in f.readlines()]
    for file in tqdm.tqdm(bol_vids_loc):
        print("Processing: ", file)
        if not os.path.exists(file.replace('video.mp4', 'openface')):
            os.mkdir(file.replace('video.mp4', 'openface'))
        os.system(f"OpenFace\FeatureExtraction.exe -f {file} -out_dir {file.replace('video.mp4', 'openface')}")

if __name__ == "__main__":
    preprocess_bol()
