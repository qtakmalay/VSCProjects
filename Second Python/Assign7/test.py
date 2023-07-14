class DualInputCNN(nn.Module):
    def __init__(self, n_in_channels: int = 1, n_hidden_layers: int = 3, n_kernels: int = 32, kernel_size: int = 7):
        super().__init__()

        # Define the first part of the network that processes the pixelated_image
        self.pixelated_image_layers = self._make_layers(n_in_channels, n_hidden_layers, n_kernels, kernel_size)

        # Define the second part of the network that processes the known_array
        self.known_array_layers = self._make_layers(n_in_channels, n_hidden_layers, n_kernels, kernel_size)

        # Define the final part of the network that combines the outputs of the previous parts
        self.combining_layers = nn.Sequential(
            nn.Conv2d(in_channels=n_kernels*2, out_channels=n_kernels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_kernels, out_channels=1, kernel_size=kernel_size, padding=kernel_size // 2)
        )

    def _make_layers(self, n_in_channels, n_hidden_layers, n_kernels, kernel_size):
        layers = []
        for i in range(n_hidden_layers):
            layers.append(nn.Conv2d(in_channels=n_in_channels, out_channels=n_kernels, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU())
            n_in_channels = n_kernels
        return nn.Sequential(*layers)

    def forward(self, pixelated_image, known_array):
        # Apply the first part of the network to the pixelated_image
        pixelated_image_out = self.pixelated_image_layers(pixelated_image)

        # Apply the second part of the network to the known_array
        known_array_out = self.known_array_layers(known_array)

        # Concatenate the outputs along the channel dimension
        combined = torch.cat([pixelated_image_out, known_array_out], dim=1)

        # Apply the final part of the network to the combined output
        predictions = self.combining_layers(combined)

        return predictions
    

with open(r"C:\Users\eljfe\OneDrive\Desktop\University\2 semester\p-i-p 2\FINAL\test_set.pkl", "rb") as f:
    data = pkl.load(f)
    pixelated_image_test_raw = data["pixelated_images"]
    known_arr_test_raw = data["known_arrays"]

combined_arr_test,known_arr_test = stack_with_padding_just_for_test(pixelated_image_test_raw,known_arr_test_raw,len(pixelated_image_test_raw))

predictions_list = []
print(f"All in all items to check: {len(combined_arr_test)}")
def step_(combined,known,device):
    with torch.no_grad():
        combined = combined.to(device=device)
        known = known.to(device=device)
        prediction = network_final(combined.view(1,2,64,64))

        actual_prediction = (torch.masked_select(prediction, known)).to(device=device)
        actual_prediction = torch.clamp(actual_prediction, min=0, max=255)
        prediction_mod = actual_prediction.detach().cpu().numpy().astype(dtype=np.uint8)
        
        predictions_list.append(prediction_mod.flatten())


        
for combined, known in tqdm(zip(combined_arr_test,known_arr_test)):
    t = step_(combined,known,device,i)

serial = serialize(predictions_list, r"C:\Users\eljfe\OneDrive\Desktop\University\2 semester\p-i-p 2\FINAL\predictions\prediction_0.txt")