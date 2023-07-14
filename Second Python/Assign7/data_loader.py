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
import submission_serialization as serialize
import utils

from tqdm import tqdm
from utils import plot
import warnings
from torch.utils.tensorboard import SummaryWriter
rng = np.random.default_rng()
width = rng.integers(0, 32, size=1)
height = rng.integers(0, 32, size=1)

batch_size = 64

os.environ['KMP_DUPLICATE_LIB_OK']='True'

train_image_path = r'C:\Users\azatv\VSCode\VSCProjects\Second Python\Assign7\training'


# def save_prediction_images(predictions, dir_path):
#     """
#     Saves the prediction images to the specified directory.
#     """
#     print("Saving prediction images...")
#     for i, prediction in enumerate(predictions):
#         # Reshape the flattened array back into its original 2D shape
#         image_array = prediction.reshape((64, 64))

#         # Create a new figure
#         plt.figure()
#         # Display the image
#         plt.imshow(image_array, cmap='gray')

#         # Generate a file path
#         file_path = os.path.join(dir_path, f'prediction_{i}.png')

#         # Save the figure to a file
#         plt.savefig(file_path)

#         # Close the figure to free up memory
#         plt.close()

def save_prediction_images(predictions, dir_path):
    print("Saving prediction images...")
    print(np.array(predictions).shape)

    for i, prediction in enumerate(predictions):
        # Calculate square root of the prediction's size to get the dimensions of the square image
        dim = int(np.sqrt(prediction.shape[0]))

        # Try reshaping to a square image
        try:
            image_array = prediction.reshape((dim, dim))
        except ValueError:
            print(f"Couldn't reshape prediction {i}, skipping...")
            continue

        # Create a new figure
        plt.figure()
        # Display the image
        plt.imshow(image_array, cmap='gray')

        # Generate a file path
        file_path = os.path.join(dir_path, f'prediction_{i}.png')

        # Save the figure to a file
        plt.savefig(file_path)

        # Close the figure to free up memory
        plt.close()

# def evaluate_model(model: torch.nn.Module, loader: torch.utils.data.DataLoader, 
#                    loss_fn, device: torch.device, prediction_path: str):
#     """
#     Function for evaluation of a model on the data in the dataloader.
#     """
#     model.eval()
#     loss = 0
#     predictions = []
#     with torch.no_grad():
#         for data in tqdm(loader, desc="Evaluating", position=0, leave=False):
#             # Get a sample and move inputs and targets to device
#             pixelated, known, targets = data
#             pixelated = pixelated.to(device)
#             known = known.to(device)
#             targets = targets.to(device)

#             # Get outputs of the specified model
#             outputs = model(pixelated, known)

#             # Apply mask to output and convert to the original pixel range
#             outputs_masked = outputs * (1 - known)

#             # Convert to uint8
#             outputs_uint8 = (outputs_masked.cpu().numpy() * 255).astype(np.uint8)

#             utils.plot_preds(outputs_uint8, r"C:\Users\azatv\VSCode\VSCProjects\Second Python\Assign7\results\preds")

#             # Flatten and append to the predictions
#             predictions.append(outputs_uint8.flatten())

#             # Add the current loss
#             loss += loss_fn(outputs, targets).item()

#     loss /= len(loader)
#     model.train()

#     # Serialize and save the predictions
#     serialize.serialize(predictions, prediction_path)

#     return loss

# def evaluate_model(model: torch.nn.Module, loader: torch.utils.data.DataLoader, 
#                    loss_fn, device: torch.device, prediction_path: str):
#     """
#     Function for evaluation of a model on the data in the dataloader.
#     """
#     model.eval()
#     loss = 0
#     predictions = []
#     with torch.no_grad():
#         for data in tqdm(loader, desc="Evaluating", position=0, leave=False):
#             # Get a sample and move inputs and targets to device
#             pixelated, known, targets = data
#             pixelated = pixelated.to(device)
#             known = known.to(device)
#             targets = targets.to(device)

#             # Get outputs of the specified model
#             outputs = model(pixelated, known)

#             # Apply mask to output to obtain only the depixelated parts
#             depixelated_output = outputs * (1 - known)

#             # Convert to uint8
#             depixelated_output_uint8 = (depixelated_output.cpu().numpy() * 255).astype(np.uint8)

#             # Flatten and append to the predictions
#             predictions.append(depixelated_output_uint8.flatten())

#             # Calculate loss only on the depixelated parts
#             loss += loss_fn(depixelated_output, targets * (1 - known)).item()

#     loss /= len(loader)
#     model.train()

#     # Serialize and save the predictions
#     serialize.serialize(predictions, prediction_path)

#     return loss




def evaluate_model_test(model: torch.nn.Module, loader: torch.utils.data.DataLoader, 
                   loss_fn, device: torch.device, prediction_path: str):
    """
    Function for evaluation of a model on the data in the dataloader.
    """
    results_path = r"C:\\Users\\azatv\\VSCode\\VSCProjects\\Second Python\\Assign7\\results"
    plot_path = os.path.join(results_path, "plots_test")
    os.makedirs(plot_path, exist_ok=True)
    model.eval()
    loss = 0
    depixelated_outputs = []
    locations = []
    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating", position=0, leave=False):
            # Get a sample and move inputs and targets to device
            if len(data) == 2:
                pixelated, known = data
                pixelated = pixelated.to(device)
                known = known.to(device).float()
                targets = None
            else:
                pixelated, known, targets = data
                pixelated = pixelated.to(device)
                known = known.to(device).float()
                targets = targets.to(device)
            

            # Get outputs of the specified model
            outputs = model(pixelated, known)
            plot(pixelated.detach().cpu().numpy(), pixelated.detach().cpu().numpy(), outputs.detach().cpu().numpy(),
                     plot_path, update=0)
            # Convert known back to boolean for the mask
            known_bool = known.bool()
            another_one = torch.masked_select(outputs, ~known_bool)
            print(another_one.size())
            print(outputs.size())
            # Apply mask to output to obtain only the depixelated parts
            depixelated_output = torch.masked_select(outputs, ~known_bool)

            # Keep track of the depixelated outputs and their locations
            depixelated_outputs.append((depixelated_output.cpu().numpy() * 255).astype(np.uint8))
            #locations.append(torch.nonzero(~known_bool, as_tuple=False).cpu().numpy())
            # Calculate loss only on the depixelated parts
            #loss += loss_fn(depixelated_output, torch.masked_select(targets, ~known_bool)).item()

    #loss /= len(loader)
    model.train()
    # for i, (depixelated, locs) in enumerate(zip(depixelated_outputs, locations)):
    #     reconstructed = utils.reconstruct_image(depixelated, locs, pixelated.shape)
    #     utils.plot_preds(reconstructed, os.path.join(prediction_path, f"{i:07d}.png"))

    # Serialize and save the depixelated outputs and their locations
    serialize.serialize(depixelated_outputs, prediction_path)

    return loss

def evaluate_model_for_train(model: torch.nn.Module, loader: torch.utils.data.DataLoader, loss_fn, device: torch.device):
    """Function for evaluation of a model ``model`` on the data in
    ``dataloader`` on device `device`, using the specified ``loss_fn`` loss
    function."""
    model.eval()
    loss = 0

    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating", position=0, leave=False):
            # Get a sample and move inputs and targets to device
            inputs1, inputs2, targets = data
            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)
            targets = targets.to(device)

            outputs = model(inputs1, inputs2)

            # Add the current loss
            loss += loss_fn(outputs, targets).item()

    loss /= len(loader)
    model.train()

    # Serialize and save the predictions
    #serialize.serialize(predictions, prediction_path)

    return loss

# def reconstruct_image(depixelated, locations, image_shape):
#     """Reconstructs a depixelated image from the given outputs and locations."""
#     batch_size, channels, height, width = image_shape
#     image = np.full((batch_size, height, width), None)  # Initialize to None
#     print(f"Reconstructing image with shape {image.shape}")
#     for pixel, loc in zip(depixelated, locations):
#         print(f"Placing pixel at location {loc} with value {pixel}")
#         # Ignore batch and channel indices
#         image[loc[0], loc[2], loc[3]] = pixel
#     # Turn remaining None values into white pixels
#     image[np.where(image == None)] = 255
#     return image


# def evaluate_model(model: torch.nn.Module, loader: torch.utils.data.DataLoader, 
#                    loss_fn, device: torch.device, prediction_path: str):
#     """
#     Function for evaluation of a model on the data in the dataloader.
#     """
#     model.eval()
#     loss = 0
#     predictions = []
#     with torch.no_grad():
#         for data in tqdm(loader, desc="Evaluating", position=0, leave=False):
#             # Get a sample and move inputs and targets to device
#             pixelated, known, targets = data
#             pixelated = pixelated.to(device)
#             known = known.to(device)
#             targets = targets.to(device)
            
#             # Get outputs of the specified model
#             outputs = model(pixelated, known)
#             known = known.bool()
#             outputs = (outputs.cpu().numpy() * 255).astype(np.uint8)
#             known = known.cpu().numpy()
            
#             depixelated_arr = outputs[~known]
#             predictions.append(depixelated_arr)
            
#             # Calculate loss only on the depixelated parts
#             #loss += loss_fn(depixelated_output, targets).item()

#     #loss /= len(loader)
#     model.train()

#     # Combine the list of numpy arrays into a single numpy array
#     predictions = np.concatenate(predictions, axis=0)

#     # Serialize and save the predictions
#     serialize.serialize(predictions, prediction_path)

#     return loss




def main(
        results_path,
        network_config: dict,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        n_updates: int = 50_000,
        device: str = "cuda"
):
    """Main function that takes hyperparameters and performs training and
    evaluation of model"""
    device = torch.device(device)
    if "cuda" in device.type and not torch.cuda.is_available():
        warnings.warn("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    
    # Set a known random seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Prepare a path to plot to
    plot_path = os.path.join(results_path, "plots")
    os.makedirs(plot_path, exist_ok=True)

    with open(r'C:\Users\azatv\VSCode\VSCProjects\Second Python\Assign7\test_set_ones.pkl', 'rb') as f:
        data = pkl.load(f)
    pix_img_np = np.array([i for i in data["pixelated_images"]])
    known_img_np = np.array([i for i in data["known_arrays"]])
    test_dataset = utils.TestDataset(pix_img_np, known_img_np)
    actual_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=utils.stack_with_padding_for_test)

    
    # Load or download CIFAR10 dataset
    dataset = utils.RandomImagePixelationDataset(train_image_path, (4, 32), (4, 32), (4, 16), dtype=np.uint8)
  
    
    # Split dataset into training, validation and test set (CIFAR10 dataset
    # is already randomized, so we do not necessarily have to shuffle again)
    training_set = torch.utils.data.Subset(
        dataset,
        indices=np.arange(int(len(dataset) * (3 / 5)))
    )
    validation_set = torch.utils.data.Subset(
        dataset,
        indices=np.arange(int(len(dataset) * (3 / 5)), int(len(dataset) * (4 / 5)))
    )
    test_set = torch.utils.data.Subset(
        dataset,
        indices=np.arange(int(len(dataset) * (4 / 5)), len(dataset))
    )
    
    # Create data sets and data loaders with rotated targets without
    # augmentation (for evaluation)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=64, shuffle=False, num_workers=0, collate_fn=utils.stack_with_padding)
    val_loader = torch.utils.data.DataLoader(validation_set, batch_size=64, shuffle=False, num_workers=0, collate_fn=utils.stack_with_padding)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0, collate_fn=utils.stack_with_padding)
 
    # Define a TensorBoard summary writer that writes to directory
    # "results_path/tensorboard"
    writer = SummaryWriter(log_dir=os.path.join(results_path, "tensorboard"))
    
    # Create Network
    net = utils.DualInputCNN(
        input_channels=1,  # Set to 1 for grayscale images
        hidden_channels=32,
        num_hidden_layers=3,
        use_batch_normalization=True,
        num_classes=10
    )
    net.to(device)
    
    # Get mse loss function
    mse = torch.nn.MSELoss()
    
    # Get adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    write_stats_at = 100  # Write status to TensorBoard every x updates
    plot_at = 1000  # Plot every x updates
    validate_at = 10  # Evaluate model on validation set and check for new best model every x updates
    update = 0  # Current update counter
    best_validation_loss = np.inf  # Best validation loss so far
    update_progress_bar = tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)
    
    # Save initial model as "best" model (will be overwritten later)
    saved_model_file = os.path.join(results_path, "best_model.pt")
    torch.save(net, saved_model_file)

    while update < n_updates:
        for data in train_loader:
            # Get next samples
            pixelated_image, known_array, targets = data

            pixelated_image = pixelated_image.to(device)
            known_array = known_array.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            outputs = net(pixelated_image, known_array)
            # Calculate loss, do backward pass and update weights
            loss = mse(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Write current training status
            if (update + 1) % write_stats_at == 0:
                writer.add_scalar(tag="Loss/training", scalar_value=loss.cpu(), global_step=update)
                for i, (name, param) in enumerate(net.named_parameters()):
                    writer.add_histogram(tag=f"Parameters/[{i}] {name}", values=param.cpu(), global_step=update)
                    writer.add_histogram(tag=f"Gradients/[{i}] {name}", values=param.grad.cpu(), global_step=update)
            
            # Plot output
            if (update + 1) % plot_at == 0:
                plot(pixelated_image.detach().cpu().numpy(), targets.detach().cpu().numpy(), outputs.detach().cpu().numpy(),
                     plot_path, update)
            
            # Evaluate model on validation set
            if (update + 1) % validate_at == 0:
                val_loss = evaluate_model_for_train(net, loader=test_loader, loss_fn=mse, device=device)
                writer.add_scalar(tag="Loss/validation", scalar_value=val_loss, global_step=update)
                # Save best model for early stopping
                if val_loss < best_validation_loss:
                    best_validation_loss = val_loss
                    torch.save(net, saved_model_file)
            
            update_progress_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
            update_progress_bar.update()

            update += 1
            if update >= n_updates:
                break
    
    update_progress_bar.close()
    writer.close()
    print("Finished Training!")
    # ser_preds = np.array([serialize.deserialize(r"C:\Users\azatv\VSCode\VSCProjects\Second Python\Assign7\results\datafiles\my_targets.data")], dtype=np.uint8)
    # save_prediction_images(ser_preds, r'C:\Users\azatv\VSCode\VSCProjects\Second Python\Assign7\results\preds')
    # print(ser_preds)

    # image = ser_preds.reshape((64, 64))

    # # Create a new matplotlib figure
    # plt.figure(figsize=(5,5))

    # # Display the image
    # plt.imshow(image, cmap='gray')

    # # Show the plot
    # plt.show()

    # Load best model and compute score on test set
    print(f"Computing scores for best model")
    net = torch.load(saved_model_file)
    # train_loss = evaluate_model_for_train(net, loader=train_loader, loss_fn=mse, device=device)
    # val_loss = evaluate_model_for_train(net, loader=val_loader, loss_fn=mse, device=device)
    # test_loss = evaluate_model_for_train(net, loader=test_loader, loss_fn=mse, device=device)
    
    actual_test_loss = evaluate_model_test(net, loader=actual_test_loader, loss_fn=mse, device=device, prediction_path=r'C:\Users\azatv\VSCode\VSCProjects\Second Python\Assign7\results\datafiles\my_targets.data')

    # print(f"Scores:")
    # print(f"  training loss: {train_loss}")
    # print(f"validation loss: {val_loss}")
    # print(f"      test loss: {test_loss}")
    
    # # Write result to file
    # with open(os.path.join(results_path, "results.txt"), "w") as rf:
    #     print(f"Scores:", file=rf)
    #     print(f"  training loss: {train_loss}", file=rf)
    #     print(f"validation loss: {val_loss}", file=rf)
    #     print(f"      test loss: {test_loss}", file=rf)


if __name__ == "__main__":
    import argparse
    import json


    # ser_preds = serialize.deserialize(r"C:\Users\azatv\VSCode\VSCProjects\Second Python\Assign7\results\datafiles\targets_debug.data")
    # test_ser_preds = serialize.deserialize(r"C:\Users\azatv\VSCode\VSCProjects\Second Python\Assign7\results\datafiles\my_targets.data")
    # print(ser_preds.shape)
    # print(test_ser_preds.shape)




    #save_prediction_images(ser_preds, r'C:\Users\azatv\VSCode\VSCProjects\Second Python\Assign7\results\preds')
    # parser = argparse.ArgumentParser()
    # parser.add_argument("config_file", type=str, help="Path to JSON config file.")
    # args = parser.parse_args()
    config_file = r"C:\Users\azatv\VSCode\VSCProjects\Second Python\Assign7\working_config.json"
    with open(config_file) as cf:
        config = json.load(cf)
    main(**config)



