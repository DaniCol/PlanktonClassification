import torch
import torch.nn as nn

def train_one_epoch(model, loader, f_loss, optimizer, device):
    """Train the model for one epoch 

    Args:
        model (torch.nn.module): the architecture of the network
        loader (torch.utils.data.DataLoader): pytorch loader containing the data
        f_loss (torch.nn.module): Cross_entropy loss for classification
        optimizer (torch.optim.Optimzer object): adam optimizer
        device (torch.device): cuda 

    Returns:
        tot_loss/N (float) : accumulated loss over one epoch 
        correct/N (float) : accuracy over one epoch
    """

    model.train()

    N = 0
    tot_loss, correct = 0.0, 0.0

    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Compute the forward pass through the network up to the loss
        outputs = model(inputs)
        loss = f_loss(outputs, targets)

        N += inputs.shape[0]
        tot_loss += inputs.shape[0] * f_loss(outputs, targets).item()
        predicted_targets = outputs.argmax(dim=1)
        correct += (predicted_targets == targets).sum().item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return tot_loss/N, correct/N

