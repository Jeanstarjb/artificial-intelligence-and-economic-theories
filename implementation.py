import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network to model supply and demand dynamics
class SupplyDemandNN(nn.Module):
    def __init__(self):
        super(SupplyDemandNN, self).__init__()
        self.fc1 = nn.Linear(2, 16)  # Input: [price, external_factor], Output: 16 hidden units
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 2)  # Output: [supply, demand]

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Generate synthetic data for supply and demand
def generate_data(num_samples=1000):
    prices = np.random.uniform(1, 100, num_samples)
    external_factors = np.random.uniform(0, 1, num_samples)
    supply = 50 + 0.5 * prices - 0.3 * external_factors * prices
    demand = 100 - 0.7 * prices + 0.2 * external_factors * prices
    data = np.stack([prices, external_factors], axis=1)
    labels = np.stack([supply, demand], axis=1)
    return data.astype(np.float32), labels.astype(np.float32)

# Train the neural network
def train_model(model, data, labels, epochs=1000, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    data_tensor = torch.tensor(data)
    labels_tensor = torch.tensor(labels)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(data_tensor)
        loss = criterion(outputs, labels_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Test the model with new data
def test_model(model, test_data):
    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(test_data)
        predictions = model(test_tensor)
    return predictions.numpy()

if __name__ == '__main__':
    # Generate training data
    train_data, train_labels = generate_data(num_samples=1000)

    # Initialize and train the model
    model = SupplyDemandNN()
    train_model(model, train_data, train_labels, epochs=1000, lr=0.001)

    # Generate test data
    test_data, _ = generate_data(num_samples=10)

    # Test the model
    predictions = test_model(model, test_data)

    # Print test results
    for i, (price, external_factor) in enumerate(test_data):
        supply, demand = predictions[i]
        print(f"Price: {price:.2f}, External Factor: {external_factor:.2f} -> Predicted Supply: {supply:.2f}, Predicted Demand: {demand:.2f}")