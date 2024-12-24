import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
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

def nn_train_time():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Generate training data
    state = np.linspace(-1, 1, 32).astype(np.float32)  # Input values
    action = np.linspace(-1, 1, 32).astype(np.float32)  # Input values

    # Convert numpy arrays to PyTorch tensors
    state = torch.tensor(state).unsqueeze(1)  # Add a dimension for compatibility
    action = torch.tensor(action).unsqueeze(1)

    model = SimpleNN()
    model.eval()

    start_time = time.time()
    predictions = model(state, action)
    end_time = time.time()
    elapsed_time = end_time - start_time
    # print(f"predictions: {predictions}")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print("nn ran")
    # import ipdb;ipdb.set_trace()



class Optimizer():
    def __init__(self, model):
        self.model = model
    # Define the objective function to minimize

    def objective_minimize(self, state, action):
        # Convert the input to a PyTorch tensor
        action_tensor = torch.tensor(action, dtype=torch.float32).reshape(-1, 1)   # Add batch dimension
        state_tensor = torch.tensor(state, dtype=torch.float32).repeat(action_tensor.shape[0], 1)  # Add batch dimension
        with torch.no_grad():
            # Predict the output for the given input
            y = self.model(state_tensor, action_tensor)
            return -y.item()  # Return the scalar value of the prediction

    # Define the constrained optimization process
    def minimize_constrained(self, fixed_state_value, num_guesses=25):
        # Ask user for the state to optimize for
        fixed_state_value = float(0.9700)
        fixed_state_tensor = torch.tensor([[fixed_state_value]], dtype=torch.float32)  # Shape (1, 1)

        # Generate initial guesses for the action variable
        initial_guesses = np.random.uniform(-1, 1, num_guesses)  # Array of initial guesses for action

        # Define bounds for the action variable
        bounds = [(-1, 1)]  # Action bounds

        # Best result tracking
        best_res = None
        best_val = float('inf')
        # Optimization loop over initial guesses
        for guess_idx, initial_guess in enumerate(initial_guesses):
            try:
                # Perform the optimization
                res = minimize(
                        lambda action: self.objective_minimize(fixed_state_tensor, action),
                        initial_guess,
                        method='trust-constr',  # Optimization method
                        jac='2-point',         # Gradient approximation
                        bounds=bounds)
                print(f"Guess {guess_idx}: x={res.x}, z={res.fun}")
                
                # Update the best result if the current result is better
                if res.fun < best_val:
                    best_res = res
                    best_val = res.fun
            except Exception as e:
                print(f"Optimization failed for guess {guess_idx}: {e}")
        
        return best_res, fixed_state_value
    
def optimizer_train_time():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = SimpleNN()
    model.eval()

    optimizer = Optimizer(model)

    start_time = time.time()
    result, state = optimizer.minimize_constrained(fixed_state_value=-0.27)
    end_time = time.time()
    elapsed_time = end_time - start_time

    if result:
        print(f"Best inputs (action): {result.x}")
        print(f"State: {state}")
        print(f"Minimum output (z): {result.fun}")
    else:
        print("Optimization failed to find a solution.")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print("optimizer ran")
    # import ipdb;ipdb.set_trace()


# nn_train_time()
optimizer_train_time()



