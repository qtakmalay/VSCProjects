
from a4_ex1 import SimpleNetwork
from dataset import get_dataset
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
def training_loop(
                    network: torch.nn.Module,
                    train_data: torch.utils.data.Dataset,
                    eval_data: torch.utils.data.Dataset,
                    num_epochs: int,
                    show_progress: bool = False
                    ) -> tuple[list, list]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network = network.to(device)  
    
    loss_function = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.SGD(network.parameters(), lr=0.1)

    batch_size = 32

    train_losses = []
    eval_losses = []

    training_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, num_workers=0)
    eval_loader = DataLoader(eval_data, shuffle=False, batch_size=batch_size, num_workers=0)
    
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    for epoch in tqdm(range(num_epochs), disable = not show_progress):
        network.train()
        train_loss = 0
        for data_sub, tar_sub in training_loader:
            data_sub, tar_sub = data_sub.float().to(device), tar_sub.float().to(device)
            optimizer.zero_grad()
            # Compute the output
            output = network(data_sub).squeeze()
            # Compute the main loss
            main_loss = loss_function(output, tar_sub)
            # Compute the gradients
            main_loss.backward()
            # Preform the update
            optimizer.step()
            
            train_loss += main_loss.item()
        train_loss /= len(training_loader)
        train_losses.append(train_loss)


        network.eval()
        with torch.no_grad():
            val_loss = 0
            for data_sub, tar_sub in eval_loader:
                # Compute the output
                data_sub, tar_sub = data_sub.float().to(device), tar_sub.float().to(device)
                output = network(data_sub).squeeze()
                # Compute the main loss
                main_loss = loss_function(output, tar_sub)
                
                val_loss += main_loss.item()
            val_loss /= len(eval_loader)
            eval_losses.append(val_loss)
    
    return train_losses, eval_losses          

if __name__ == "__main__":
    torch.random.manual_seed(0)
    train_data, eval_data = get_dataset()
    network = SimpleNetwork(32, 128, 1)
    train_losses, eval_losses = training_loop(network, train_data, eval_data, num_epochs=10, show_progress=True)
    for epoch, (tl, el) in enumerate(zip(train_losses, eval_losses)):
        print(f"Epoch: {epoch} --- Train loss: {tl:7.2f} --- Eval loss: {el:7.2f}")