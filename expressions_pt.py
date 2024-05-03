import torch                                        
import torch.utils
from torch.utils.data import Dataset, DataLoader    
import torch.nn as nn                               
import glob 
import tqdm 
import pandas as pd
from torchinfo import summary 
from timeit import default_timer as timer 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from utils import *
import warnings 
warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using", device)

                               
class ExpressionDataset(Dataset):
    
    def __init__(self, paths, transform=None):
        self.paths = []
        for path in paths:
            self.paths += glob.glob(path + "run_*/expressions.txt")
        self.annotations = pd.read_csv("BagOfLies/Annotations.csv")
        self.vid_annotations = self.annotations["video"].str.replace("./Finalised", "BagOfLies/Finalised").str.replace("/video.mp4", "")
        self.transform = transform
        self.max_seq_len = 1265

    def load_expressions(self, file_path):
        df = pd.read_csv(file_path)
        le = LabelEncoder()
        df = le.fit_transform(df)
        df = torch.tensor(df).unsqueeze(1)
        return torch.tensor(df)
    
    def get_ground_truth(self, file_path):
        video_run = "/".join(file_path.split("/")[:-1])
        ind = self.vid_annotations[self.vid_annotations == video_run].index
        return self.annotations.iloc[ind]["truth"].values[0]
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        
        X = self.load_expressions(self.paths[index]).to(device).to(torch.float32)
        y = self.get_ground_truth(self.paths[index]).astype(torch.LongTensor)
        if self.transform:
            return self.transform(X), y 
        else:
            return X, y # return data, label (X, y)

class LandmarkLSTM(nn.Module):  
    def __init__(self, input_shape, hidden_shape, output_shape):
        super().__init__()
        self.lstm = nn.LSTM(input_shape, hidden_shape, batch_first=True)

        self.dense = nn.Sequential(
            nn.Linear(hidden_shape, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, output_shape),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor):
        out, _ = self.lstm(x)
        out = self.dense(out[:, -1])
        print(out.shape)
        return out
    
def collate_pad(batch):
    X = [item[0] for item in batch]
    y = [item[1] for item in batch]
    X = nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=0)
    return X, torch.tensor(y)

paths = glob.glob("BagOfLies/Finalised/User_*/")
train_paths, test_paths = torch.utils.data.random_split(paths, [int(0.8*len(paths)), len(paths) - int(0.8*len(paths))])

train_dataset = ExpressionDataset(train_paths)
test_dataset = ExpressionDataset(test_paths)

print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(test_dataset)}")
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_pad)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, collate_fn=collate_pad)

X_batch, y_batch = next(iter(train_dataloader))
X_single, y_single = X_batch[0], y_batch[0]
X_single = X_single.unsqueeze(0)
print(f"X_batch shape: {X_batch.shape}")
print(f"y_batch shape: {y_batch.shape}")

model = LandmarkLSTM(1, 64, 2).to(device)

model.eval()

with torch.inference_mode():
    output = model(X_single)
    print(f"Testing model with input shape {X_single.shape}...")
    print(output)

    summary(model, input_size=(X_single.shape), device=device)


NUM_EPOCHS = 50

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

start_time = timer()

# Train model_0 
model_results = train(model=model, 
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn, 
                        epochs=NUM_EPOCHS)

# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")

plot_loss_curves(model_results)
plt.savefig('outputs/Expressions_LSTM')