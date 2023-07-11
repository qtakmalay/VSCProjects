import pickle as pkl
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
from PIL import Image
import numpy as np
import os, glob, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import utils

rng = np.random.default_rng()
width = rng.integers(0, 32, size=1)
height = rng.integers(0, 32, size=1)

batch_size = 64

os.environ['KMP_DUPLICATE_LIB_OK']='True'

train_image_path = r'C:\Users\azatv\VSCode\VSCProjects\Second Python\Assign7\training'

'''
Each test set image was transformed to a shape of (64, 64) using transforms.Resize(shape=64,
interpolation=InterpolationMode.BILINEAR) followed by transforms.CenterCrop(size=(64,
64)) and converted to grayscale before the pixelation process (see Assignment 2 – Exercise 2) was
applied. In this pixelation process, for each image, the x- and y-coordinates were randomly chosen
from the ranges [0, 64 − width] and [0, 64 − height], where width and height were again randomly
chosen from from the ranges [4, 32] and [4, 32]. The pixelation block size was randomly chosen from
the range [4, 16] (see also Assignment 3 – Exercise 1).
Your task is to predict the true, original values for each of the pixelated_images and collect the
predictions in a list as follows:
• Each prediction must be a flat 1D NumPy array of data type np.uint8 containing all predicted
pixel values in a flattened view (e.g., such as obtained via some_image[~known_array]).
• The order of this list must be the same as the order of the lists in the test set.
The points are determined based on the model performance, the root-mean-squared error (RMSE),
which is compared to two reference models: IdentityModel (uses input as prediction, i.e., it does
not perform any actual computation) and BasicCNNModel (5 layers with 32 kernels of size 3). If your
model’s RMSE is equal to or higher than IdentityModel, you get 0 points. If your model’s RMSE
is equal to BasicCNNModel, you get 200 points. Everything in between the two models is linearly
interpolated. If your model’s RMSE is lower than BasicCNNModel, you get 50 bonus points. There
is also a third model (only relevant for bonus points): AdvancedModel. If you manage to beat this
as well, you instead get 100 bonus points. See the leaderboard for the individual RMSE scores.
General Project Hints:
• Divide the project into subtasks and check your program regularly. Example:
1. Decide which samples you want to use in your training, validation or test sets.
2. Create the data loader and stacking function for the minibatches (see code files of Unit 5).
3. Implement a neural network (NN) that computes an output given this input (see code
files of Unit 6).
4. Implement the computation of the loss between NN output and target (see code files of
Unit 7).
5. Implement the NN training loop (see code files of Unit 7),
6. Implement the evaluation of the model performance on a validation set (see code files of
Unit 7).
You can use the project structure shown in the example project in the code files of Unit 7.
• You only have 5 attempts to submit predictions, so it will be important for you to use some
samples for a validation set and maybe another test set to get an estimate for the generalization
of your model.
• It makes sense to only work with inputs of sizes that can appear in the test set. For this, you
can use the torchvision transforms, e.g., as follows:
from torchvision import transforms
from PIL import Image
im_shape = 64
resize_transforms = transforms.Compose([
transforms.Resize(size=im_shape),
transforms.CenterCrop(size=(im_shape, im_shape)),
])
with Image.open(filename) as image:
image = resize_transforms(image)
However, if you want to drastically increase your data set size using data augmentation, you
can also use random cropping followed by resizing to create more input images (e.g., via
transforms.RandomResizedCrop). For more details on data augmentation, see Unit 8.
• You will most likely need to write a stacking function for the DataLoader (collate_fn). For
this, you can take the maximum over the X and the maximum over the Y dimensions of the
input array and create a zero-tensor of shape (n_samples, n_feature_channels, max_X,
max_Y), so that it can hold the stacked input arrays. Then you can copy the input values into
this zero-tensor. However, if you know that your input already only has a certain shape (see
resizing transformation above), you can directly stack.
• It makes sense to feed additional input into the NN. You can concatenate the channels of
pixelated_image and known_array (see Assignment 2 – Exercise 2) and feed the resulting
tensor as input into the network.
• Creating predictions and computing loss: To predict the unknown pixel values, you can implement
a CNN that creates an output that has the same size as the input (see code files of
Unit 7). Then you can use either a boolean mask like known_array or slicing to obtain the
predicted pixel values.
• If you normalize the NN input, denormalize the NN output accordingly if you want to predict
a non-normalized target array. The challenge inputs and targets will not be normalized. You
do not have access to the targets to normalize them, so you will need to create non-normalized
predictions. In practice, this might be done by saving the mean and variance you used for
normalizing the input and using these values to denormalize the NN output.
• Start with a small data subset to quickly get suitable hyperparameters. Use a small subset (e.g.,
30 samples or at the very beginning even just 2 samples) of your training set for debugging.
Your model should be able to overfit (=achieve almost perfect performance) on such a small
data set.
• Debug properly. Check the inputs and outputs of your network manually. Do not trust
that everything works just because the loss decreases. Debug with num_workers=0 for the
DataLoader to be able to step into the data loading process.
• To show how the predictions file should look like, a debug predictions file is available (download).
These predictions are just 0-value NumPy arrays, so do not use them for any other
purpose than debugging. Use the deserialize function of the provided submission_serialization.
py script to deserialize the file into its original list of NumPy arrays.
• You do not need to reinvent the wheel. Most of this project can be solved by reusing parts of
the code materials and assignments from this semester.
• When uploading to the challenge server, you might not immediately see a result due to the
server’s task scheduling. Please be patient and check back later. This also means that you
should try to upload your prediction in due time to avoid having no immediate feedback when
the assignment deadline approaches.

'''
# class BasicCNN(nn.Module):
#     def __init__(
#             self,
#             input_channels: int,
#             hidden_channels: int,
#             num_hidden_layers: int,
#             use_batch_normalization: bool,
#             num_classes: int,
#             kernel_size: int = 3,
#             activation_function: nn.Module = nn.ReLU()
#     ):
#         super().__init__()
#         self.input_channels = input_channels
#         self.hidden_channels = hidden_channels
#         self.num_hidden_layers = num_hidden_layers
#         self.use_batch_normalization = use_batch_normalization
#         self.num_classes = num_classes
#         self.kernel_size = kernel_size
#         self.activation_function = activation_function
        
#         self.conv_layers = nn.ModuleList()
#         self.conv_layers.append(nn.Conv2d(
#             input_channels,
#             hidden_channels,
#             kernel_size,
#             padding="same",
#             padding_mode="zeros",
#             bias=not use_batch_normalization
#         ))
#         # We already added one conv layer, so start the range from 1 instead of 0
#         for i in range(1, num_hidden_layers):
#             self.conv_layers.append(nn.Conv2d(
#                 hidden_channels,
#                 hidden_channels,
#                 kernel_size,
#                 padding="same",
#                 padding_mode="zeros",
#                 bias=not use_batch_normalization
#             ))
#         if self.use_batch_normalization:
#             self.batch_norm_layers = nn.ModuleList()
#             for i in range(num_hidden_layers):
#                 self.batch_norm_layers.append(nn.BatchNorm2d(hidden_channels))
#         self.output_layer = nn.Linear(self.hidden_channels * 64 * 64, self.num_classes)
    
#     def forward(self, input_images: torch.Tensor) -> torch.Tensor:
#         for i in range(self.num_hidden_layers):
#             input_images = self.conv_layers[i](input_images)
#             if self.use_batch_normalization:
#                 input_images = self.batch_norm_layers[i](input_images)
#             input_images = self.activation_function(input_images)
#         input_images = input_images.view(-1, self.hidden_channels * 64 * 64)
#         input_images = self.output_layer(input_images)
#         return input_images
    

# class BasicCNN(nn.Module):
#     def __init__(self):
#         super(BasicCNN, self).__init__()
#         self.conv1 = nn.Conv2d(2, 32, 3, padding=1)  # input channels = 2 (pixelated_image and known_array)
#         self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
#         self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
#         self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
#         self.conv5 = nn.Conv2d(32, 1, 3, padding=1)  # output channels = 1 (grayscale image)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         x = self.conv5(x)  # No activation function on the last layer
#         return x


# class BasicCNN(nn.Module):
#     def __init__(self):
#         super(BasicCNN, self).__init__()

#         # The architecture consists of 5 convolutional layers
#         # Each layer followed by a ReLU activation and Batch Normalization
#         self.layers = nn.Sequential(
#             nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(32),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(32),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(32),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(32),
#             nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         # Forward pass through the layers
#         return self.layers(x)

class SimpleCNN(torch.nn.Module):
    
    def __init__(self, n_in_channels: int = 1, n_hidden_layers: int = 3, n_kernels: int = 32, kernel_size: int = 7):
        """Simple CNN with ``n_hidden_layers``, ``n_kernels`` and
        ``kernel_size`` as hyperparameters."""
        super().__init__()
        
        cnn = []
        for i in range(n_hidden_layers):
            cnn.append(torch.nn.Conv2d(
                in_channels=n_in_channels,
                out_channels=n_kernels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ))
            cnn.append(torch.nn.ReLU())
            n_in_channels = n_kernels
        self.hidden_layers = torch.nn.Sequential(*cnn)
        
        self.output_layer = torch.nn.Conv2d(
            in_channels=n_in_channels,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
    
    def forward(self, x):
        """Apply CNN to input ``x`` of shape ``(N, n_channels, X, Y)``, where
        ``N=n_samples`` and ``X``, ``Y`` are spatial dimensions."""
        # Apply hidden layers: (N, n_in_channels, X, Y) -> (N, n_kernels, X, Y)
        cnn_out = self.hidden_layers(x)
        # Apply output layer: (N, n_kernels, X, Y) -> (N, 1, X, Y)
        predictions = self.output_layer(cnn_out)
        return predictions


def collate_fn(batch):
    # batch is a list of data points, each data point being a tuple of 
    # (pixelated_image, known_array, target_array)

    # unzip the batch to get separate lists of pixelated_images, known_arrays, and target_arrays
    pixelated_images, known_arrays, target_arrays = zip(*batch)
    
    # compute maximum X and Y dimensions
    max_X = max(img.shape[-2] for img in pixelated_images)
    max_Y = max(img.shape[-1] for img in pixelated_images)
    
    # create zero tensors
    batched_pixelated_images = torch.zeros(len(batch), 1, max_X, max_Y)
    batched_known_arrays = torch.zeros(len(batch), 1, max_X, max_Y)
    batched_target_arrays = torch.zeros(len(batch), 1, max_X, max_Y)
    
    # copy data into zero tensors
    for i in range(len(batch)):
        X, Y = pixelated_images[i].shape[-2], pixelated_images[i].shape[-1]
        batched_pixelated_images[i, :, :X, :Y] = pixelated_images[i]
        batched_known_arrays[i, :, :X, :Y] = known_arrays[i]
        batched_target_arrays[i, :, :X, :Y] = target_arrays[i]
    
    return batched_pixelated_images, batched_known_arrays, batched_target_arrays




with open(r'C:\Users\azatv\VSCode\VSCProjects\Second Python\Assign7\test_set_ones.pkl', 'rb') as f:
    data = pkl.load(f)
pix_img_np = np.array([i for i in data["pixelated_images"]])
known_img_np = np.array([i for i in data["known_arrays"]])



train_dataset = utils.RandomImagePixelationDataset(train_image_path, (4, 32), (4, 32), (4, 16), dtype=np.uint8)


train_size = int(0.8 * len(train_dataset))
valid_size = len(train_dataset) - train_size

train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])

#test loader
test_dataset = utils.TestDataset(pix_img_np, known_img_np)
test_loader = DataLoader(test_dataset, 
                         batch_size=batch_size, 
                         shuffle=False, 
                         num_workers=4)

#train loader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=utils.stack_with_padding)

#validation loader
valid_loader = DataLoader(valid_dataset, 
                          batch_size=batch_size, 
                          shuffle=False, collate_fn=utils.stack_with_padding)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Initialize the model
model = SimpleCNN(2, 3, 32, 3).to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Number of epochs
epochs = 10

# Training loop
for epoch in range(epochs):
    model.train()
    for pixelated_images, known_arrays, target_arrays, _ in train_loader:
        # Combine pixelated_images and known_arrays as channels and move to device
        inputs = torch.cat((pixelated_images.unsqueeze(1), known_arrays.unsqueeze(1)), dim=1).to(device)
        targets = target_arrays.to(device)

        # Forward pass
        outputs = model(inputs.squeeze(2))
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for pixelated_images, known_arrays, target_arrays, _ in valid_loader:
            # Combine pixelated_images and known_arrays as channels and move to device
            inputs = torch.cat((pixelated_images.unsqueeze(1), known_arrays.unsqueeze(1)), dim=1).to(device)
            targets = target_arrays.to(device)

            # Forward pass and loss computation
            outputs = model(inputs.squeeze(2))
            loss = criterion(outputs, targets)

            total_loss += loss.item()

        print(f"Validation Loss: {total_loss / len(valid_loader)}")



# pixelated_image, known_array, target_array, image_file = train_dataset[0]
# pixelated_image = Image.fromarray(np.squeeze(pixelated_image).astype(np.uint8))
# #known_array = np.where(known_img_np[0], 255, 0).astype(np.uint8)
# target_array = Image.fromarray(np.squeeze(pixelated_image).astype(np.uint8))
# known_array = Image.fromarray(np.squeeze(known_array).astype(np.uint8))

# fig, axes = plt.subplots(6, 3, figsize=(15, 30))  # Increase the figure size for visibility

# for i in range(6):  # Adjust this to the number of images you want
#     pixelated_image, known_array, target_array, _ = train_dataset[i]
#     axes[i, 0].imshow(pixelated_image, cmap='gray')
#     axes[i, 0].set_title(f'Pixelated Image {i+1}')
#     axes[i, 1].imshow(known_array, cmap='gray')
#     axes[i, 1].set_title(f'Known Array {i+1}')
#     axes[i, 2].imshow(target_array, cmap='gray')
#     axes[i, 2].set_title(f'Target Array {i+1}')

# # Remove the axis labels for clean plots
# for ax in axes.ravel():
#     ax.axis('off')

# fig.tight_layout()
# plt.show()

