import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
from tqdm import tqdm

def training_loop(
    network: torch.nn.Module,
    train_data: torch.utils.data.Dataset,
    eval_data: torch.utils.data.Dataset,
    loss,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    #momentum: float
) -> tuple[list, list]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = network.to(device)

    #optimizer = torch.optim.SGD(params=network.parameters(), lr=learning_rate, momentum=momentum)
    optimizer = Adam(network.parameters(), lr=learning_rate)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False)
    loss_function = loss

    train_losses = []
    eval_losses = []
    
    lowest_loss = float('inf')
    best_model_path = "model.pth"
    
    for epoch in range(num_epochs):
        
        network.train()
        epoch_train_loss = 0.0
        
        train_loader = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")

        for batch in train_loader:
            features, targets = batch
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            output = network(features)
            loss = loss_function(output.squeeze(dim=1), targets)
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            average_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(average_train_loss)

        network.eval()
        epoch_eval_loss = 0.0

        eval_loader = tqdm(eval_loader, desc=f"Evaluating Epoch {epoch+1}/{num_epochs}")

        with torch.no_grad():
            for batch in eval_loader:
                features, targets = batch
                features, targets = features.to(device), targets.to(device)
                output = network(features)
                loss = loss_function(output, targets)
                epoch_eval_loss += loss.item()
                if loss.item() < lowest_loss:
                    lowest_loss = loss.item()
                    torch.save(network.state_dict(), best_model_path)

        average_eval_loss = epoch_eval_loss / len(eval_loader)
        eval_losses.append(average_eval_loss)
        
        print(f'Epoch {epoch + 1:2d} finished with training loss: {train_losses[-1]:.6f}' +
              (f' and validation loss: {eval_losses[-1]:.6f}' if eval_losses else ''))

    return train_losses, eval_losses
    