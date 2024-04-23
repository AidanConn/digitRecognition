import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from PIL import Image, ImageOps  # Import ImageOps module

# Load the data
data = []
labels = []
for file in os.listdir("data"):
    img = Image.open("data/" + file)
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))

    # Invert colors
    img = ImageOps.invert(img)  # Use ImageOps to invert colors

    img = np.array(img)
    img = img / 255.0  # Normalize the image
    data.append(img)
    label = int(file.split(".")[0].split('-')[0])  # Extract the label from the filename
    labels.append(label)
    print("Loaded:", file)

data = np.array(data)
labels = np.array(labels)

# Print the shape of the data
print(data.shape)
print(labels.shape)

# Show what the data looks like for all the number have 8 versions
import matplotlib.pyplot as plt
for i in range(10):
    for j in range(8):
        plt.subplot(10, 8, i * 8 + j + 1)
        plt.imshow(data[i * 8 + j], cmap='gray')
        plt.axis('off')
plt.show()



# PyTorch model to predict the number in the image
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Create model instance
model = Model()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert data and labels to PyTorch tensors
data = torch.tensor(data, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)

# Reshape data for convolutional input
data = data.view(-1, 1, 28, 28)

# Train the model
num_epochs = 20
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the model
torch.save(model.state_dict(), "model.pth")

# Load the model
model = Model()
model.load_state_dict(torch.load("model.pth"))


# Test the model
model.eval()
correct = 0
with torch.no_grad():
    for i in range(len(data)):
        output = model(data[i].view(1, 1, 28, 28))
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted = torch.argmax(probabilities, dim=1).item()
        prediction_percentage = probabilities[0][predicted].item() * 100
        actual = labels[i].item()
        print(f"Prediction: {predicted}, Actual: {actual}, Prediction Confidence: {prediction_percentage:.2f}%")
        if predicted == actual:
            correct += 1

print(f"Accuracy: {correct / len(data) * 100:.2f}%")


# Test every image in the test folder to see what number it is and show the image
for file in os.listdir("test"):
    img = Image.open("test/" + file)
    img = img.convert('L')
    img = img.resize((28, 28))
    img = ImageOps.invert(img)
    img = np.array(img)
    img = img / 255.0
    img = torch.tensor(img, dtype=torch.float32)
    img = img.view(1, 1, 28, 28)
    model.eval()
    with torch.no_grad():
        output = model(img)
        predicted = torch.argmax(output, dim=1).item()
        print(f"Prediction: {predicted}")
        plt.imshow(img.view(28, 28), cmap='gray')
        plt.title(f"Prediction: {predicted}")
        plt.show()




