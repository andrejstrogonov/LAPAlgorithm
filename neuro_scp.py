import pandas as pd
import numpy as np
import numba as nb
import torch
from torch import nn

# Use numba to optimize the check_rule function
@nb.njit
def check_rule(transactions, rule, min_support):
    support = len([transaction for transaction in transactions if rule.issubset(transaction)]) / len(transactions)
    return support >= min_support

# Define a PyTorch model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Train the PyTorch model
def train_model(X, y):
    model = NeuralNetwork()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    X = torch.tensor(X.values).float()
    y = torch.tensor(y.values).float().view(-1, 1)

    for epoch in range(100):
        outputs = model(X)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model

# Predict association rules
def predict_rules(model, X):
    predictions = model(torch.tensor(X.values).float())
    predictions = predictions.detach().numpy()

    result = []
    for i in range(len(transactions)):
        for j in range(i+1, len(transactions)):
            rule = set(transactions[i]) | set(transactions[j])
            if check_rule(transactions, rule, min_support):
                result.append((transactions[i], transactions[j]))

    return result

# Main function
def main():
    # Convert transactions to a suitable format for the neural network
    X = pd.DataFrame({'item': list(transactions.keys()), 'count': list(transactions.values())})
    X['count'] = X['count'].astype('int')  # Convert values in the count column to integer type

    # Create a one-hot encoding for the 'item' column
    X = pd.get_dummies(X, columns=['item'])

    # Generate random stochastic data
    stochastic_data = np.random.randn(len(X))
    X['stochastic'] = stochastic_data

    # Train the PyTorch model
    model = train_model(X.drop(columns=['count']), X[['count']])

    # Use the trained model to predict association rules
    rules = predict_rules(model, X.drop(columns=['count']))

    print("Rules:")
    for rule in rules:
        print("- " + str(rule))

if __name__ == "__main__":
    main()