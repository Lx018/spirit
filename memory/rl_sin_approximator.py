
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. Define the Agent (our function approximator)
class FunctionApproximator(nn.Module):
    def __init__(self):
        super(FunctionApproximator, self).__init__()
        self.layer1 = nn.Linear(1, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# 2. Set up the Environment and Hyperparameters
agent = FunctionApproximator()
optimizer = optim.Adam(agent.parameters(), lr=0.001)
training_epochs = 10000
batch_size = 64

# 3. The Training Loop
for epoch in range(training_epochs):
    # --- Environment: Get a batch of states (random x values) ---
    x_values = torch.rand(batch_size, 1) * 10 - 5  # Random x between -5 and 5

    # --- Agent: Take actions (predict y values) ---
    predicted_y = agent(x_values)

    # --- Environment: Get the true y values for reward calculation ---
    true_y = torch.sin(x_values)

    # --- Environment: Calculate the reward ---
    # Reward is higher for smaller errors. We use negative absolute error.
    reward = -torch.abs(predicted_y - true_y)

    # --- Learning: Update the agent based on the reward ---
    # Our "loss" is the negative of the reward.
    # By minimizing the loss, we maximize the reward.
    loss = -reward.mean()

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch [{epoch+1}/{training_epochs}], Loss: {loss.item():.4f}")

# 4. Visualize the Results
print("\nTraining finished. Visualizing results...")
x_test = torch.linspace(-5, 5, 100).view(-1, 1)
predicted_y_test = agent(x_test).detach().numpy()
true_y_test = torch.sin(x_test).numpy()

plt.figure(figsize=(10, 6))
plt.plot(x_test.numpy(), true_y_test, label="True sin(x)", color="blue")
plt.plot(x_test.numpy(), predicted_y_test, label="RL Agent's Approximation", color="red", linestyle="--")
plt.title("Reinforcement Learning: Approximating sin(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
