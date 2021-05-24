import torch
from torch import nn
import numpy as np
import torchvision
from torchvision import transforms, datasets
import pandas as pd
import os
from skimage import io, transform
from torch.utils.data import DataLoader




class ShipsDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = pd.read_csv(csv_file)

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):



        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.labels.iloc[idx, 0])
        image = io.imread(img_name)
        label = self.labels.iloc[idx, 1]
        
        sample = {'image': image, 'label': label}

        if self.transform:
            try: 
                sample["image"] = self.transform(sample["image"])
            except: 
                temp_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((100, 100)), transforms.Normalize((0.5,), (0.5,))])
                sample["image"] = temp_transform(sample["image"])
        
        return sample




transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((100, 100)), transforms.Grayscale(num_output_channels=1), transforms.Normalize((0.5,), (0.5,))])
ships_training_data = ShipsDataset(csv_file='./ship_data/train/train.csv', root_dir='./ship_data/train/images', transform=transform)
ships_training_data = DataLoader(ships_training_data, batch_size=64)




class NeuralNetwork(nn.Module): 
    def __init__(self): 
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(100*100, 512), 
            nn.ReLU(), 
            nn.Linear(512, 128), 
            nn.ReLU(), 
            nn.Linear(128, 5), 
            nn.ReLU()
        )

    def forward(self, x): 

        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001)






for i in range(10): 
    err = 0
    print("epoch" + str(i))

    for batch, data in enumerate(ships_training_data): 

        pred = model(data["image"])
        #print(pred)
    
        loss = loss_fn(pred, torch.tensor(data["label"] - 1)) / len(data["image"])
        err = err + loss.item()
        #print(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



        if (batch % 10 == 0): 
            print("batch: " + str(batch) + " / " + str(len(ships_training_data)))

    print("error: " + str(err))
    
    



correct = 0
total = 0

for batch, data in enumerate(ships_training_data): 

    for i in range(len(data["image"])): 

        pred = model(data["image"][i])

        if (pred.argmax() == data["label"][i]): 
            correct = correct + 1

        total = total + 1

print("accuracy: ")
print(correct/total)