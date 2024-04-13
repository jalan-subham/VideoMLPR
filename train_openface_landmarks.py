import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf 
import glob 
import pandas as pd

files = glob.glob("BagOfLies/Data/User_*/run_*/openface/*.csv")   
print(files)
data_load_count = 0
sample_csv = pd.read_csv(files[0])
columns = sample_csv.columns.to_list()
start_ind = columns.index(" X_0")
end_ind = columns.index(" Z_67")

annotations = pd.read_csv("BagOfLies/Annotations.csv")
video_annotations = annotations["video"].str.replace("./Finalised", "BagOfLies/Data").str.replace("/video.mp4", "")
vid_lens = []
def get_ground_truth(file_path):
    video_run = "/".join(file_path.split("/")[:-2])
    ind = video_annotations[video_annotations == video_run].index
    return annotations.iloc[ind]["truth"].values[0]

def load_face_landmarks_truth(index):
    global data_load_count, vid_lens
    df = pd.read_csv(files[data_load_count])
    print(f"Parsing {files[data_load_count]}")
    data_load_count += 1
    print(data_load_count)
    output_X = df.iloc[:, start_ind:end_ind + 1].to_numpy()
    vid_lens.append(output_X.shape[0])
    output_y = get_ground_truth(files[data_load_count])
    return output_X, output_y

train_ds = tf.data.Dataset.from_tensor_slices(files).map(load_face_landmarks_truth).batch(32)






