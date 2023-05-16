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
        """CNN, consisting of ``num_hidden_layers`` linear layers, using relu
        activation function in the hidden CNN layers.
        
        Parameters
        ----------
        input_channels: int -- input_channels
            Number of features channels in input tensor
            hidden_channels: specifies the number of feature channels
        num_hidden_layers: int -- num_hidden_layers
            Number of hidden layers

        use_batch_normalization controls whether 2-dimensional batch normalization is used

        num_classes specifies the number of output neurons of the fully-connected

        activation_function specifies which non-linear activation function is used

        n_hidden_kernels: int -- kernel_size
            Number of kernels in each hidden layer
        n_output_channels: int
            Number of features in output tensor
        """
        super().__init__()
        
        hidden_layers = []
        for _ in range(num_hidden_layers):
            # Add a CNN layer
            layer = torch.nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels, kernel_size= kernel_size//2)
            hidden_layers.append(layer)
            # Add relu activation module to list of modules
            hidden_layers.append(activation_function)
            if use_batchnormalization: hidden_layers.append(torch.nn.BatchNorm2d(hidden_channels))
            input_channels = hidden_channels
        
        self.hidden_layers = torch.nn.Sequential(*hidden_layers)
        self.output_layer = torch.nn.Conv2d(in_channels=input_channels, out_channels=num_classes, kernel_size=kernel_size)
    
    def forward(self, input_images: torch.Tensor):
        """Apply CNN to ``x``.
        
        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape ``(n_samples, input_channels, height, width)``
        
        Returns
        ----------
        torch.Tensor
            Output tensor of shape ``(n_samples, n_output_channels, h', w')``
        """

        # Apply hidden layers module
        hidden_features = self.hidden_layers(input_images)
        #hidden_features[1]
        # Apply last layer (=output layer)
        output = self.output_layer(hidden_features)
        output = torch.Tensor.flatten(start_dim=4, end_dim=2)
        return output




if __name__ == "__main__":
    torch.random.manual_seed(0)
    network = SimpleCNN(3, 32, 3, True, 10, activation_function=torch.nn.ELU())
    input = torch.randn(1, 3, 64, 64)
    output = network(input)
    print(output)