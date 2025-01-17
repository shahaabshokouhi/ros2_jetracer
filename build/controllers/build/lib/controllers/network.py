import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# class Actor(nn.Module):
#     def __init__(self, grid_size):
#         super(Actor, self).__init__()
#         n_input = grid_size * grid_size + 1
#         self.fc1 = nn.Linear(n_input,64)
#         self.fc2 = nn.Linear(64, 128)
#         self.fc3 = nn.Linear(128, 256)
#         self.fc4 = nn.Linear(256, 128)
#         self.fc5 = nn.Linear(128, 64)
#         self.fc6 = nn.Linear(64, 32)
#         self.fc7 = nn.Linear(32, 1)
#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()

#     def forward(self, x):
#         if isinstance(x, np.ndarray):
#             x = torch.tensor(x, dtype=torch.float)
#         if isinstance(x, tuple):
#             x = tuple(torch.tensor(item, dtype=torch.float) for item in x)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         x = self.relu(x)
#         x = self.fc4(x)
#         x = self.relu(x)
#         x = self.fc5(x)
#         x = self.relu(x)
#         x = self.fc6(x)
#         x = self.relu(x)
#         x = self.fc7(x)
#         x = self.tanh(x)
#         x = x * torch.pi

#         return x

class Actor(nn.Module):
    def __init__(self, grid_size):
        super(Actor, self).__init__()
        n_input = grid_size * grid_size + 1
        self.fc1 = nn.Linear(n_input, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        if isinstance(x, tuple):
            x = tuple(torch.tensor(item, dtype=torch.float) for item in x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.fc3(x)
        x = self.leaky_relu(x)
        x = self.fc4(x)
        x = self.tanh(x)
        x = x * torch.pi / 2.0
        return x
    
# actor with attention mechanism
# class Actor(nn.Module):
#     def __init__(self, grid_size):
#         super(Actor, self).__init__()
#         self.grid_size = grid_size
#         n_grid_inputs = grid_size * grid_size
#         n_angle_input = 1

#         # Grid input processing
#         self.grid_fc1 = nn.Linear(n_grid_inputs, 64)
#         self.grid_fc2 = nn.Linear(64, 128)
#         self.grid_fc3 = nn.Linear(128, 64)

#         # Angle input processing
#         self.angle_fc1 = nn.Linear(n_angle_input, 32)
#         self.angle_fc2 = nn.Linear(32, 32)

#         # Attention mechanism
#         self.attention_fc = nn.Linear(64, 1)  # From grid features to attention score
#         self.sigmoid = nn.Sigmoid()

#         # Combined processing
#         self.combined_fc1 = nn.Linear(64 + 32, 64)
#         self.combined_fc2 = nn.Linear(64, 64)
#         self.output_layer = nn.Linear(64, 1)

#         # Activation functions
#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
#         self.tanh = nn.Tanh()

#     def forward(self, x):
#         # Split the inputs
#         if isinstance(x, np.ndarray):
#             x = torch.tensor(x, dtype=torch.float)
#         if isinstance(x, tuple):
#             x = tuple(torch.tensor(item, dtype=torch.float) for item in x)
#         elif isinstance(x, torch.Tensor):
#             pass  # x is already a tensor
#         else:
#             x = torch.tensor(x, dtype=torch.float)

#         # Ensure x is the correct shape
#         if x.dim() == 1:
#             x = x.unsqueeze(0)

#         grid_inputs = x[:, :self.grid_size * self.grid_size]  # Inputs 1-100
#         angle_input = x[:, self.grid_size * self.grid_size:]  # Input 101

#         # Process grid inputs
#         grid_features = self.leaky_relu(self.grid_fc1(grid_inputs))
#         grid_features = self.leaky_relu(self.grid_fc2(grid_features))
#         grid_features = self.leaky_relu(self.grid_fc3(grid_features))

#         # Compute attention score from grid features
#         attention_score = self.sigmoid(self.attention_fc(grid_features))  # Shape: [batch_size, 1]

#         # Process angle input
#         angle_features = self.leaky_relu(self.angle_fc1(angle_input))
#         angle_features = self.leaky_relu(self.angle_fc2(angle_features))

#         # Apply attention to angle features
#         attended_angle = angle_features * attention_score  # Element-wise multiplication

#         # Combine grid features and attended angle features
#         combined_features = torch.cat((grid_features, attended_angle), dim=1)

#         # Further processing
#         x = self.leaky_relu(self.combined_fc1(combined_features))
#         x = self.leaky_relu(self.combined_fc2(x))

#         # Output layer
#         x = self.output_layer(x)
#         x = self.tanh(x)
#         x = x * torch.pi  # Scale output to [-π, π]

#         return x

class Critic(nn.Module):
    def __init__(self, grid_size):
        super(Critic, self).__init__()
        n_input = grid_size * grid_size + 1
        self.fc1 = nn.Linear(n_input, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        if isinstance(x, tuple):
            x = tuple(torch.tensor(item, dtype=torch.float) for item in x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        # x = self.tanh(x)

        return x