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
            nn.Linear(16, 32),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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
    transactions = {'A': 5, 'B': 3, 'C': 2, 'D': 4, 'E': 1}
    min_support = 0.2

    # Convert transactions to a suitable format for the neural network
    X = pd.DataFrame({'item': list(transactions.keys()), 'count': list(transactions.values())})
    X['count'] = X['count'].astype('int')  # Convert values in the count column to integer type

    # Add stochastic data with NumPy
    stochastic_data = np.random.randn(len(X))
    X['stochastic'] = stochastic_data

    # Train the PyTorch model
    model = train_model(X[['count', 'stochastic']], X[['count']])

    # Use the trained model to predict association rules
    rules = predict_rules(model, X[['count', 'stochastic']])

    print("Rules:")
    for rule in rules:
        print("- " + str(rule))

if __name__ == "__main__":
    main()
