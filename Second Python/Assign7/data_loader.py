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


def save_prediction_images(predictions, dir_path):
    print("Saving prediction images...")
    print(np.array(predictions).shape)
    #print(predictions)
    for i, prediction in enumerate(predictions):
        # Reshape the flattened array back into its original 2D shape
                # Squeeze unnecessary dimensions
        #print(predictions)
        print(predictions)
        prediction = prediction.squeeze()
        
        # Reshape the flattened array back into its original 2D shape
        image_array = prediction.reshape((64, 64))

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

def evaluate_model(model: torch.nn.Module, loader: torch.utils.data.DataLoader, loss_fn, device: torch.device, prediction_path: str):
    """Function for evaluation of a model ``model`` on the data in
    ``dataloader`` on device `device`, using the specified ``loss_fn`` loss
    function."""
    model.eval()
    loss = 0
    predictions = []
    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating", position=0, leave=False):
            # Get a sample and move inputs and targets to device
            inputs1, inputs2, targets = data
            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)
            targets = targets.to(device)
            
            # # Get outputs of the specified model
            # outputs = model(inputs1, inputs2)
            # #print((outputs.cpu().numpy() * 255).shape)
            # outputs_uint8 = (outputs.cpu().numpy() * 255).astype(np.uint8).flatten()


            outputs = model(inputs1, inputs2)
    
            # Apply mask to output
            outputs_masked = outputs * (1 - inputs2) + inputs1 * inputs2
            print(outputs_masked.shape)
            # Convert to uint8
            outputs_uint8 = (outputs_masked.cpu().numpy() * 255).astype(np.uint8).flatten()
            
            #predictions.append([outputs_uint8])

            predictions.append(outputs_uint8)
            # Add the current loss
            loss += loss_fn(outputs, targets).item()

    loss /= len(loader)
    model.train()

    # Serialize and save the predictions
    #serialize.serialize(predictions, prediction_path)

    return loss




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
    print(net)
    net.to(device)
    
    # Get mse loss function
    mse = torch.nn.MSELoss()
    
    # Get adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    write_stats_at = 100  # Write status to TensorBoard every x updates
    plot_at = 1000  # Plot every x updates
    validate_at = 50  # Evaluate model on validation set and check for new best model every x updates
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
            print(known_array)
            optimizer.zero_grad()

            outputs = net(pixelated_image, known_array)
            print(known_array.size())
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
                val_loss = evaluate_model(net, loader=val_loader, loss_fn=mse, device=device, prediction_path=r'C:\Users\azatv\VSCode\VSCProjects\Second Python\Assign7\results\targets.data')
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
    # ser_preds = np.array([serialize.deserialize(r"C:\Users\azatv\VSCode\VSCProjects\Second Python\Assign7\results\targets_debug.data")], dtype=np.uint8)
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
    # train_loss = evaluate_model(net, loader=train_loader, loss_fn=mse, device=device)
    # val_loss = evaluate_model(net, loader=val_loader, loss_fn=mse, device=device)
    # test_loss = evaluate_model(net, loader=test_loader, loss_fn=mse, device=device)
    
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
    print("Finished Training!")
    # No need to wrap it with np.array()
    ser_preds = serialize.deserialize(r"C:\Users\azatv\VSCode\VSCProjects\Second Python\Assign7\results\targets_debug.data")
    save_prediction_images(ser_preds, r'C:\Users\azatv\VSCode\VSCProjects\Second Python\Assign7\results\preds')
    # parser = argparse.ArgumentParser()
    # parser.add_argument("config_file", type=str, help="Path to JSON config file.")
    # args = parser.parse_args()
    config_file = r"C:\Users\azatv\VSCode\VSCProjects\Second Python\Assign7\working_config.json"
    with open(config_file) as cf:
        config = json.load(cf)
    main(**config)



