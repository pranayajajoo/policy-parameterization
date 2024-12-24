import time
import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize


# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden1 = nn.Linear(2, 64)  # First hidden layer
        self.hidden2 = nn.Linear(64, 64)  # Second hidden layer
        self.output = nn.Linear(64, 1)   # Output layer
        self.sigmoid = nn.Sigmoid()      # Activation function

    def forward(self, state, action):
        inputs_train = torch.cat([state, action], dim=1)
        x = self.sigmoid(self.hidden1(inputs_train))  # Apply sigmoid after first hidden layer
        x = self.sigmoid(self.hidden2(x))  # Apply sigmoid after second hidden layer
        x = self.output(x)                 # No activation on the output layer
        return x


class Optimizer:
    def __init__(self, model):
        self.model = model

    def objective_minimize(self, state, action):
        """
        Objective function for optimization. Evaluates the model for given state and action.
        """
        action_tensor = torch.tensor(action, dtype=torch.float32).reshape(-1, 1)   # Add batch dimension
        state_tensor = torch.tensor(state, dtype=torch.float32).repeat(action_tensor.shape[0], 1)  # Repeat state
        with torch.no_grad():
            y = self.model(state_tensor, action_tensor)
            return -y.numpy()  # Return array of values for batch processing

    def minimize_parallel(self, states):
        """
        Finds the best action for each state in the batch using parallel optimization.
        Uses two guesses: the last best guess (if available) and a new random guess.
        """
        num_states = states.shape[0]
        action_dim = 1  # Dimensionality of actions
        bounds = [(-1, 1)]  # Action bounds for all states
        results = []

        # Generate two guesses for each state
        initial_guesses1 = np.random.uniform(-1, 1, (num_states, action_dim))  # Random guesses
        initial_guesses2 = np.random.uniform(-1, 1, (num_states, action_dim))  # Another random guess

        for i, state in enumerate(states):
            state_tensor = state.unsqueeze(0)  # Ensure shape is [1, state_dim]
            best_val = float('inf')
            best_action = None

            for guess in [initial_guesses1[i], initial_guesses2[i]]:
                res = minimize(
                    lambda action: self.objective_minimize(state_tensor, action),
                    guess,
                    method='trust-constr',
                    jac='2-point',
                    bounds=bounds
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_action = res.x

            results.append((state.cpu().numpy(), best_action, -best_val))  # Save state, action, and predicted value

        return results


def main():
    # Set up device and initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = SimpleNN().to(device)
    model.eval()

    # Generate 32 states
    states = torch.linspace(-1, 1, 32).unsqueeze(1).to(device)  # Shape: [32, 1]

    # Generate corresponding actions from the model
    with torch.no_grad():
        actions_nn = model(states, torch.zeros_like(states))  # Actions from NN

    # Save states and NN actions
    saved_data = {"states": states.cpu().numpy(), "actions_nn": actions_nn.cpu().numpy()}
    print("Saved states and actions from NN.")

    # Run optimization
    optimizer = Optimizer(model)
    results = optimizer.minimize_parallel(states)

    # Compare optimized actions with NN actions
    for idx, (state, optimized_action, optimized_value) in enumerate(results):
        nn_action = actions_nn[idx].item()
        print(f"State: {state}, NN Action: {nn_action:.4f}, Optimized Action: {optimized_action[0]:.4f}, Value: {optimized_value:.4f}")

    print("Optimization and comparison complete.")


if __name__ == "__main__":
    main()
