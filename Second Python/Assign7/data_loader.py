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
        count = 0
        for data in tqdm(loader, desc="Evaluating", position=0, leave=False):

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

            outputs = model(pixelated, known)

            count += 1
            known_bool = known.bool()

            depixelated_output = torch.masked_select(outputs, ~known_bool)

            depixelated_outputs.append((depixelated_output.cpu().numpy() * 255).astype(np.uint8))

    model.train()

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
    

    np.random.seed(0)
    torch.manual_seed(0)
    

    plot_path = os.path.join(results_path, "plots")
    os.makedirs(plot_path, exist_ok=True)

    with open(r'C:\Users\azatv\VSCode\VSCProjects\Second Python\Assign7\test_set_ones.pkl', 'rb') as f:
        data = pkl.load(f)
    pix_img_np = np.array([i for i in data["pixelated_images"]])
    known_img_np = np.array([i for i in data["known_arrays"]])
    test_dataset = utils.TestDataset(pix_img_np, known_img_np)
    actual_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=utils.stack_with_padding_for_test)

    

    dataset = utils.RandomImagePixelationDataset(train_image_path, (4, 32), (4, 32), (4, 16), dtype=np.uint8)
  

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
    

    train_loader = torch.utils.data.DataLoader(training_set, batch_size=64, shuffle=False, num_workers=0, collate_fn=utils.stack_with_padding)
    val_loader = torch.utils.data.DataLoader(validation_set, batch_size=64, shuffle=False, num_workers=0, collate_fn=utils.stack_with_padding)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0, collate_fn=utils.stack_with_padding)
 

    writer = SummaryWriter(log_dir=os.path.join(results_path, "tensorboard"))
    
    
    net = utils.DualInputCNN(
        input_channels=1,  
        hidden_channels=32,
        num_hidden_layers=3,
        use_batch_normalization=True,
        num_classes=10
    )
    net.to(device)
    

    mse = torch.nn.MSELoss()
    

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    write_stats_at = 100 
    plot_at = 1000 
    validate_at = 1000
    update = 0 
    best_validation_loss = np.inf  
    update_progress_bar = tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)
    depixelated_outputs = []

    saved_model_file = os.path.join(results_path, "best_model.pt")
    torch.save(net, saved_model_file)

    while update < n_updates:
        for data in train_loader:

            pixelated_image, known_array, targets = data

            pixelated_image = pixelated_image.to(device)
            known_array = known_array.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            outputs = net(pixelated_image, known_array)
       
            loss = mse(outputs, targets)
            loss.backward()
            optimizer.step()
            
            known_bool = known_array.bool()
            #another_one = torch.masked_select(outputs, ~known_bool)
            #print(another_one.size())
            #print(outputs.size())
            depixelated_output = torch.masked_select(outputs, ~known_bool)

            depixelated_outputs.append((depixelated_output.cpu().detach().numpy() * 255).astype(np.uint8))
            if (update + 1) % write_stats_at == 0:
                writer.add_scalar(tag="Loss/training", scalar_value=loss.cpu(), global_step=update)
                for i, (name, param) in enumerate(net.named_parameters()):
                    writer.add_histogram(tag=f"Parameters/[{i}] {name}", values=param.cpu(), global_step=update)
                    writer.add_histogram(tag=f"Gradients/[{i}] {name}", values=param.grad.cpu(), global_step=update)
            

            if (update + 1) % plot_at == 0:
                plot(pixelated_image.detach().cpu().numpy(), targets.detach().cpu().numpy(), outputs.detach().cpu().numpy(),
                     plot_path, update)

            if (update + 1) % validate_at == 0:
                evaluate_model_test(net, loader=actual_test_loader, loss_fn=mse, device=device, prediction_path=r'C:\Users\azatv\VSCode\VSCProjects\Second Python\Assign7\results\datafiles\my_targets.data')

            update_progress_bar.update()

            update += 1
            if update >= n_updates:
                break
    
    update_progress_bar.close()
    writer.close()
    print("Finished Training!")

    print(f"Computing scores for best model")
    net = torch.load(saved_model_file)

    actual_test_loss = evaluate_model_test(net, loader=actual_test_loader, loss_fn=mse, device=device, prediction_path=r'C:\Users\azatv\VSCode\VSCProjects\Second Python\Assign7\results\datafiles\my_targets.data')
    serialize.serialize(depixelated_outputs, r'C:\Users\azatv\VSCode\VSCProjects\Second Python\Assign7\results\datafiles\my_targets.data')






# def main(
#         results_path,
#         network_config: dict,
#         learning_rate: float = 1e-3,
#         weight_decay: float = 1e-5,
#         n_updates: int = 50_000,
#         device: str = "cuda"
# ):
#     device = torch.device(device)
#     if "cuda" in device.type and not torch.cuda.is_available():
#         warnings.warn("CUDA not available, falling back to CPU")
#         device = torch.device("cpu")
    
#     np.random.seed(0)
#     torch.manual_seed(0)
#     plot_path = os.path.join(results_path, "plots")
#     os.makedirs(plot_path, exist_ok=True)

#     with open(r'C:\Users\azatv\VSCode\VSCProjects\Second Python\Assign7\test_set_ones.pkl', 'rb') as f:
#         data = pkl.load(f)
#     pix_img_np = np.array([i for i in data["pixelated_images"]])
#     known_img_np = np.array([i for i in data["known_arrays"]])
#     test_dataset = utils.TestDataset(pix_img_np, known_img_np)
#     actual_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=utils.stack_with_padding_for_test)

#     dataset = utils.RandomImagePixelationDataset(train_image_path, (4, 32), (4, 32), (4, 16), dtype=np.uint8)
  
#     training_set = torch.utils.data.Subset(
#         test_dataset,
#         indices=np.arange(int(len(test_dataset) * (3 / 5)))
#     )
#     train_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=utils.stack_with_padding_for_test)
#     writer = SummaryWriter(log_dir=os.path.join(results_path, "tensorboard"))
    
   
#     net = utils.DualInputCNN(
#         input_channels=1,
#         hidden_channels=32,
#         num_hidden_layers=3,
#         use_batch_normalization=True,
#         num_classes=10
#     )
#     net.to(device)
    
#     mse = torch.nn.MSELoss()
#     depixelated_outputs = []

#     optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
#     write_stats_at = 100  
#     plot_at = 1000  
#     validate_at = 1000  
#     update = 0  
#     best_validation_loss = np.inf 
#     update_progress_bar = tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)
    
#     saved_model_file = os.path.join(results_path, "best_model.pt")
#     torch.save(net, saved_model_file)

#     while update < n_updates:
#         for data in train_loader:
#             pixelated_image, known_array = data

#             pixelated_image = pixelated_image.to(device)
#             known_array = known_array.to(device)
#             #targets = targets.to(device)
#             optimizer.zero_grad()

#             outputs = net(pixelated_image, known_array)
#             #loss = mse(outputs, targets)
#             #loss.backward()
#             #optimizer.step()
#             plot(pixelated_image.detach().cpu().numpy(), pixelated_image.detach().cpu().numpy(), outputs.detach().cpu().numpy(),
#                      plot_path, update=update)
#             #count += 1
#             known_bool = known_array.bool()
#             #another_one = torch.masked_select(outputs, ~known_bool)
#             #print(another_one.size())
#             #print(outputs.size())
#             depixelated_output = torch.masked_select(outputs, ~known_bool)

#             depixelated_outputs.append((depixelated_output.cpu().detach().numpy() * 255).astype(np.uint8))
#             # if (update + 1) % write_stats_at == 0:
#             #     writer.add_scalar(tag="Loss/training", scalar_value=loss.cpu(), global_step=update)
#             #     for i, (name, param) in enumerate(net.named_parameters()):
#             #         writer.add_histogram(tag=f"Parameters/[{i}] {name}", values=param.cpu(), global_step=update)
#             #         writer.add_histogram(tag=f"Gradients/[{i}] {name}", values=param.grad.cpu(), global_step=update)
            
#             if (update + 1) % plot_at == 0:
#                 plot(pixelated_image.detach().cpu().numpy(), known_array.detach().cpu().numpy(), outputs.detach().cpu().numpy(),
#                      plot_path, update)
            
#             if (update + 1) % validate_at == 0:
#                 val_loss = evaluate_model_test(net, loader=actual_test_loader, loss_fn=mse, device=device, prediction_path=r'C:\Users\azatv\VSCode\VSCProjects\Second Python\Assign7\results\datafiles\my_targets1.data')

#                 #writer.add_scalar(tag="Loss/validation", scalar_value=val_loss, global_step=update)
#                 # Save best model for early stopping
#                 # if val_loss < best_validation_loss:
#                 #     best_validation_loss = val_loss
#                 #     torch.save(net, saved_model_file)
            
#             #update_progress_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
#             update_progress_bar.update()

#             update += 1
#             if update >= n_updates:
#                 break
    
#     update_progress_bar.close()
#     writer.close()
#     print("Finished Training!")
#     #actual_test_loss = evaluate_model_test(net, loader=actual_test_loader, loss_fn=mse, device=device, prediction_path=r'C:\Users\azatv\VSCode\VSCProjects\Second Python\Assign7\results\datafiles\my_targets1.data')

#     # ser_preds = np.array([serialize.deserialize(r"C:\Users\azatv\VSCode\VSCProjects\Second Python\Assign7\results\datafiles\my_targets.data")], dtype=np.uint8)
#     # save_prediction_images(ser_preds, r'C:\Users\azatv\VSCode\VSCProjects\Second Python\Assign7\results\preds')
#     # print(ser_preds)

#     # image = ser_preds.reshape((64, 64))

#     # # Create a new matplotlib figure
#     # plt.figure(figsize=(5,5))

#     # # Display the image
#     # plt.imshow(image, cmap='gray')

#     # # Show the plot
#     # plt.show()

#     print(f"Computing scores for best model")
#     net = torch.load(saved_model_file)
#     train_loss = evaluate_model_for_train(net, loader=train_loader, loss_fn=mse, device=device)
#     # val_loss = evaluate_model_for_train(net, loader=val_loader, loss_fn=mse, device=device)
#     # test_loss = evaluate_model_for_train(net, loader=test_loader, loss_fn=mse, device=device)
    
    
#     # print(f"Scores:")
#     # print(f"  training loss: {train_loss}")
#     # print(f"validation loss: {val_loss}")
#     # print(f"      test loss: {test_loss}")
    
#     # # Write result to file
#     # with open(os.path.join(results_path, "results.txt"), "w") as rf:
#     #     print(f"Scores:", file=rf)
#     #     print(f"  training loss: {train_loss}", file=rf)
#     #     print(f"validation loss: {val_loss}", file=rf)
#     #     print(f"      test loss: {test_loss}", file=rf)



def main2(
        results_path,
        network_config: dict,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        n_updates: int = 50_000,
        device: str = "cuda"
):
    with open(r'C:\Users\azatv\VSCode\VSCProjects\Second Python\Assign7\test_set_ones.pkl', 'rb') as f:
            data = pkl.load(f)
    pix_img_np = np.array([i for i in data["pixelated_images"]])
    known_img_np = np.array([i for i in data["known_arrays"]])
    test_dataset = utils.TestDataset(pix_img_np, known_img_np)
    
    actual_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=utils.stack_with_padding_for_test)

    training_set = torch.utils.data.Subset(
        test_dataset,
        indices=np.arange(int(len(test_dataset) * (3 / 5)))
    )
    validation_set = torch.utils.data.Subset(
        test_dataset,
        indices=np.arange(int(len(test_dataset) * (3 / 5)), int(len(test_dataset) * (4 / 5)))
    )
    test_set = torch.utils.data.Subset(
        test_dataset,
        indices=np.arange(int(len(test_dataset) * (4 / 5)), len(test_dataset))
    )    

    BATCH_SIZE = 64  

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = utils.DualInputCNN(
        input_channels=1,  
        hidden_channels=32,
        num_hidden_layers=3,
        use_batch_normalization=True,
        num_classes=10
    )


    model_path = "C:/Users/azatv/VSCode/VSCProjects/Second Python/Assign7/results/best_model.pt"
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    loss_fn = torch.nn.MSELoss()

    for update in range(n_updates):
        train_model(model, loader=train_loader, loss_fn=loss_fn, device=device)
        if update % 1000 == 0: 
            valid_loss = evaluate_model_test(model, loader=valid_loader, loss_fn=loss_fn, device=device)
            print(f'Update: {update}, Validation loss: {valid_loss}')

    test_loss = evaluate_model_test(model, loader=test_loader, loss_fn=loss_fn, device=device, prediction_path=results_path)
    print(f'Test loss: {test_loss}')



if __name__ == "__main__":
    import argparse
    import json

    config_file = r"C:\Users\azatv\VSCode\VSCProjects\Second Python\Assign7\working_config.json"
    with open(config_file) as cf:
        config = json.load(cf)
    main(**config)



