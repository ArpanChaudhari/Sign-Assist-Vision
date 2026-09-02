import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


# 1. Load the hand gesture data

# Read the CSV file created by collect_data.py
data = pd.read_csv("data/hand_data.csv")

# Get all 63 hand landmark coordinates
X = data.drop("label", axis=1).values

# Get the gesture labels such as A, B, C, etc.
y = data["label"].values


# 2. Convert labels into numbers

# Machine learning models work with numbers, not text
encoder = LabelEncoder()

# Example: A -> 0, B -> 1, C -> 2
y_encoded = encoder.fit_transform(y)

# Get the total number of different gesture classes
num_classes = len(encoder.classes_)

# Save the class names for later use in app.py
np.save("models/classes.npy", encoder.classes_)


# 3. Convert data into PyTorch tensors

# Convert hand coordinates into float tensors
X_tensor = torch.FloatTensor(X)

# Convert labels into integer tensors
y_tensor = torch.LongTensor(y_encoded)


# Combine input data and labels into one dataset
dataset = TensorDataset(X_tensor, y_tensor)


# Create batches of 16 samples for training
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True  # shuffle=True mixes the data before each training cycle
)


# 4. Create the neural network

class SignModel(nn.Module):

    def __init__(self, num_classes):

        # Initialize the parent PyTorch class
        super().__init__()

        # Create the layers of the neural network
        self.network = nn.Sequential(

            # Input: 63 hand landmark values, Output: 128 values
            nn.Linear(63, 128),

            # Add non-linearity
            nn.ReLU(),

            # Reduce 128 values to 64
            nn.Linear(128, 64),

            # Add non-linearity
            nn.ReLU(),

            # Final layer predicts the gesture class
            nn.Linear(64, num_classes)
        )


    # This function defines how data moves through the model
    def forward(self, input_data):

        return self.network(input_data)


# Create the model
model = SignModel(num_classes)


# 5. Set the loss function and optimizer

# CrossEntropyLoss is commonly used for classification problems
criterion = nn.CrossEntropyLoss()

# Adam updates the model weights during training
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# 6. Train the model

# Number of times the model will see the full dataset
epochs = 100

print("Training started... This should only take a few seconds!")


# Repeat the training process for each epoch
for epoch in range(epochs):

    # Store the total loss for this epoch
    total_loss = 0


    # Get one batch of input data and labels at a time
    for batch_X, batch_y in dataloader:

        # Clear old gradients from the previous step
        optimizer.zero_grad()


        # Send the input data through the model
        predictions = model(batch_X)


        # Calculate how wrong the predictions are
        loss = criterion(predictions, batch_y)


        # Calculate gradients
        loss.backward()


        # Update the model weights
        optimizer.step()


        # Add the current batch loss to total loss
        total_loss += loss.item()


    # Print training progress every 10 epochs
    if (epoch + 1) % 10 == 0:

        average_loss = total_loss / len(dataloader)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {average_loss:.4f}"
        )


# 7. Save the trained model

# Save only the learned model weights
torch.save(
    model.state_dict(),
    "models/model.pth"
)


print("Model saved to model.pth!")