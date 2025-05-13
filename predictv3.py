# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 12:01:50 2024

@author: anash
"""

import pickle
import torch
import lightning as L
import numpy as np
from LSTMClassifier import LSTMClassifierTrain
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
import random

with open(r"train_data.pkl", "rb") as file:
    train_data = pickle.load(file)
    
with open(r"val_data.pkl", "rb") as file:
    val_data = pickle.load(file)
    
v_seq = torch.load(r"C:\Users\anash\OneDrive\Dokumente\voltage_measurements.pt")

def create_mask(tensor_list):
    masks = []
    for tensor in tensor_list:
        if isinstance(tensor,torch.Tensor):
            masks.append([value.item()<=200 for value in tensor])
        else:
            masks.append([])
    return masks


def filter_val(data,num):
    # Extract labels (keys) from the dictionary
    # Extract clips, features, and labels from the input dictionary
    clips = data['clips']
    features = data['features']
    labels = data['labels']
    
    # Initialize lists to store selected clips, features, and labels
    selected_clips = []
    selected_features = []
    selected_labels = []
    
    # Iterate through each label (0, 1, 2)
    for label_value in [0, 1, 2]:
        # Get the indices of entries with the current label
        indices_for_label = [i for i, label in enumerate(labels) if label == label_value]
        
        # Randomly select 30 indices from this list
        selected_indices = random.sample(indices_for_label, num)
        
        # Use the selected indices to gather corresponding clips, features, and labels
        selected_clips.extend([clips[i] for i in selected_indices])
        selected_features.extend([features[i] for i in selected_indices])
        selected_labels.extend([labels[i] for i in selected_indices])
    
    # Create the output dictionary with the same structure as the input
    output_dict = {
        'clips': selected_clips,
        'features': selected_features,
        'labels': selected_labels
    }
    
    return output_dict
    # 'filtered_data' now contains the randomly selected entries for each label
#var_filtered_val = filter_val(val_data,30)
#with open(r"var_filtered_val.pkl", "wb") as file:
#    pickle.dump(var_filtered_val, file)

voltage_train = list()
img_v_train = list()

for i, c in enumerate(train_data["clips"]):
    voltage_train.append(v_seq[c-1])
    img_v_train.append(torch.concat((train_data["features"][i], v_seq[c-1]), dim=1))
    
train_masks = create_mask(voltage_train)

voltage_val = list()
img_v_val = list()

for i, c in enumerate(val_data["clips"]):
    voltage_val.append(v_seq[c-1])
    img_v_val.append(torch.concat((val_data["features"][i], v_seq[c-1]), dim=1))
    
val_masks = create_mask(voltage_val)

train = [voltage_train[i][mask] for i, mask in enumerate(train_masks)]
val = [voltage_val[i][mask] for i, mask in enumerate(val_masks)]

train_labels = torch.tensor(train_data["labels"])
val_labels = torch.tensor(val_data["labels"])

train_packed_sequences = pack_sequence(train,enforce_sorted=False)
train_unpacked_sequences, lengths = pad_packed_sequence(train_packed_sequences, batch_first=True)
val_packed_sequences = pack_sequence(val,enforce_sorted=False)
val_unpacked_sequences, lengths = pad_packed_sequence(val_packed_sequences, batch_first=True)

train_dataset = TensorDataset(train_unpacked_sequences, train_labels) 
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataset = TensorDataset(val_unpacked_sequences, val_labels) 
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)

model = LSTMClassifierTrain(input_size=1, hidden_size=200, num_classes=3, num_layers=2)
trainer = L.Trainer(max_epochs=600, log_every_n_steps=5, enable_progress_bar=True, accelerator='gpu')
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


model.to("cuda")
count = 0
count_no_1s = 0
accuracy = 0
inaccuracy = 0
observed = []
predicted = []
print("\nAfter training:")
for idx, input1 in enumerate(val_unpacked_sequences):
        with torch.no_grad():
            prediction = model(input1.unsqueeze(0).to("cuda"))  # Füge Batch-Dimension hinzu
            observed_class = val_labels[idx].item()
            predicted_class = prediction.argmax(dim=1).item()
            observed.append(observed_class)
            predicted.append(predicted_class)
            if observed_class != 6:
                print(f"Clip {train_data["clips"][idx]:03}: Observed = {observed_class}, Predicted = {predicted_class}")
                if observed_class == predicted_class:
                    accuracy += 1
                elif np.abs(observed_class-predicted_class) == 2:
                    inaccuracy += 1
                if observed_class != 1:
                    count_no_1s +=1
                count += 1
print(f"\nPrediction accuracy: {round((accuracy/count)*100, 2)} %")
print(f"Very inaccurate predictions: {round((inaccuracy/count_no_1s)*100, 2)} %")
