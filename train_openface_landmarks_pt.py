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

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using", device)
                               

class FaceLandmarkDataset(Dataset):
    
    def __init__(self, dataset, lib = "openface", transform=None):
        if dataset =="bol":
            self.paths = glob.glob("BagOfLies/Data/User_*/run_*/openface/*.csv")
            self.annotations = pd.read_csv("BagOfLies/Annotations.csv")
            self.vid_annotations = self.annotations["video"].str.replace("./Finalised", "BagOfLies/Data").str.replace("/video.mp4", "")
        if lib == "openface":
            sample_cols = pd.read_csv(self.paths[0]).columns.to_list()
            self.start_ind = sample_cols.index(" X_0")
            self.end_ind = sample_cols.index(" Z_67")
        self.transform = transform

        # self.max_seq_len = max([self.load_landmarks(path).shape[0] for path in self.paths])
        # print(self.max_seq_len)

        self.max_seq_len = 1265

    def load_landmarks(self, file_path):
        df = pd.read_csv(file_path)
        output_X = df.iloc[:, self.start_ind:self.end_ind + 1].to_numpy()
        return output_X
    
    def get_ground_truth(self, file_path):
        video_run = "/".join(file_path.split("/")[:-2])
        ind = self.vid_annotations[self.vid_annotations == video_run].index
        return self.annotations.iloc[ind]["truth"].values[0]
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        
        X = self.load_landmarks(self.paths[index])
        y = self.get_ground_truth(self.paths[index])
        X = [torch.tensor(X), torch.zeros((self.max_seq_len, X.shape[1]))] 
        X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
        X = X[0, :, :].squeeze(0).float()

        if self.transform:
            return self.transform(X), y 
        else:
            return X, y # return data, label (X, y)

class LandmarkLSTM(nn.Module):  
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.lstm = nn.LSTM(input_shape, 4, batch_first=True)

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5060, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.Dropout(0.5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor):
        out, _ = self.lstm(x)
        out = self.dense(out)
        return out
    
# def collate_pad(batch):
#     data = [torch.from_numpy(item[0]).float() for item in batch]
#     target = [item[1] for item in batch]
#     data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
#     target = torch.tensor(target)
#     return data, target

dataset = FaceLandmarkDataset("bol")

train_set, val_set = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset) - int(0.8*len(dataset))])
print(f"Training set size: {len(train_set)}")
print(f"Validation set size: {len(val_set)}")
train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
test_dataloader = DataLoader(val_set, batch_size=16, shuffle=True)

img_batch, label_batch = next(iter(test_dataloader))
img_single, label_single = img_batch[0].to(device), label_batch[0]
img_single = img_single.unsqueeze(0)
model = LandmarkLSTM(204, 2).to(device)

model.eval()

with torch.inference_mode():
    output = model(img_single)
    print(f"Testing model with input shape {img_single.shape}...")
    print(output)

    summary(model, input_size=(img_single.shape), device=device)

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer):
    # Put model in train mode
    model.train()
    
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module):
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
    
            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            # Calculate and accumulate accuracy        
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm.tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn)
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # 6. Return the filled results at the end of the epochs
    return results

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

def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

plot_loss_curves(model_results)
plt.savefig('outputs/nonNorm_acrossUsers_LSTMDense2.png')