import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

class DSNN(torch.nn.Module):
    
    def __init__(self, n_input_features: int, n_hidden_layers: int, n_hidden_units: int, n_output_features: int):
        """Fully-connected feed-forward neural network, consisting of
        ``n_hidden_layers`` linear layers, using selu activation function in the
        hidden layers.

        Parameters
        ----------
        n_input_features: int
            Number of features in input tensor
        n_hidden_layers: int
            Number of hidden layers
        n_hidden_units: int
            Number of units in each hidden layer
        n_output_features: int
            Number of features in output tensor
        """
        super().__init__()
        
        # We want to use n_hidden_layers linear layers
        hidden_layers = []
        for _ in range(n_hidden_layers):
            # Add linear layer module to list of modules
            layer = torch.nn.Linear(in_features=n_input_features, out_features=n_hidden_units)
            torch.nn.init.normal_(layer.weight, 0, 1 / np.sqrt(layer.in_features))
            hidden_layers.append(layer)
            # Add selu activation module to list of modules
            hidden_layers.append(torch.nn.SELU())
            n_input_features = n_hidden_units
        
        self.hidden_layers = torch.nn.Sequential(*hidden_layers)
        
        # The output layer usually is separated to allow easy access to the
        # internal features (the model's data representation after the hidden
        # layers; see feature extraction example in 04_data_analysis.py)
        self.output_layer = torch.nn.Linear(in_features=n_hidden_units, out_features=n_output_features)
        torch.nn.init.normal_(self.output_layer.weight, 0, 1 / np.sqrt(self.output_layer.in_features))
    
    def forward(self, x: torch.Tensor):
        """Apply deep SNN to ``x``.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape ``(n_samples, n_input_features)`` or
            ``(n_input_features,)``

        Returns
        ----------
        torch.Tensor
            Output tensor of shape ``(n_samples, n_output_features)`` or
            ``(n_output_features,)``
        """
        # Apply hidden layers module
        hidden_features = self.hidden_layers(x)
        
        # Apply last layer (=output layer) without selu activation
        output = self.output_layer(hidden_features)
        return output


class ImprovedMyCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedMyCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 7 * 7, 120)  # Adjusted for two pooling layers
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 7 * 7)  # Adjusted for two pooling layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here, assuming using CrossEntropyLoss which includes Softmax
        return x
