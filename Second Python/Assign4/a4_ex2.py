import torch

class SimpleCNN(torch.nn.Module):
    def __init__(
    self,
    input_channels: int,
    hidden_channels: int,
    num_hidden_layers: int,
    use_batchnormalization: bool,
    num_classes: int,
    kernel_size: int = 3,
    activation_function: torch.nn.Module = torch.nn.ReLU()
    ):
        super().__init__()
        
        hidden_layers = []
        for _ in range(num_hidden_layers):
            # Add a CNN layer
            layer = torch.nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels, kernel_size= kernel_size, padding=kernel_size//2)
            hidden_layers.append(layer)
            if use_batchnormalization: 
                hidden_layers.append(torch.nn.BatchNorm2d(hidden_channels))
            # Add relu activation module to list of modules
            hidden_layers.append(activation_function)
            
            input_channels = hidden_channels
        hidden_layers.append(torch.nn.AdaptiveAvgPool2d((1, 1)))
        self.conv = torch.nn.Sequential(*hidden_layers)
        # in_features: int,
        # out_features: int,
        self.output_layer = torch.nn.Linear(in_features=input_channels, out_features=num_classes)
    
    def forward(self, input_images: torch.Tensor):
        """Apply CNN to ``input_images``.
        
        Parameters
        ----------
        input_images: torch.Tensor
            Input tensor of shape ``(n_samples, input_channels, height, width)``
        
        Returns
        ----------
        torch.Tensor
            Output tensor of shape ``(n_samples, n_output_channels, h', w')``
        """

        # Apply hidden layers module
        x = self.conv(input_images)
        x = x.view(x.size(0), -1)
        x = self.output_layer(x)
        return x




if __name__ == "__main__":
    torch.random.manual_seed(0)
    network = SimpleCNN(3, 32, 3, True, 10, activation_function=torch.nn.ELU())
    input = torch.randn(1, 3, 64, 64)
    output = network(input)
    print(output)