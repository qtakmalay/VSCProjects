import torch

class SimpleNetwork(torch.nn.Module):
    def __init__(self, input_neurons: int, hidden_neurons: int, output_neurons: int, activation_function: torch.nn.Module = torch.nn.ReLU()):
        super().__init__()

        self.layer_0 = torch.nn.Linear(in_features=input_neurons, out_features=hidden_neurons)
        self.layer_1 = torch.nn.Linear(in_features=hidden_neurons, out_features=hidden_neurons)
        self.layer_2 = torch.nn.Linear(in_features=hidden_neurons, out_features=hidden_neurons)
        self.layer_3 = torch.nn.Linear(in_features=hidden_neurons, out_features=output_neurons)

        self.activation_function = activation_function
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # N is the minibatch size and F is the number of features per sample
        x = self.layer_0(x)
        x = self.activation_function(x)
        x = self.layer_1(x)
        x = self.activation_function(x)
        x = self.layer_2(x)
        x = self.activation_function(x)
        output = self.layer_3(x)
        return output



if __name__ == "__main__":
    torch.random.manual_seed(0)
    simple_network = SimpleNetwork(10, 20, 5)
    
    input = torch.randn(1, 10)
    output = simple_network(input)
    print(output)
