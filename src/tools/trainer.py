"""This module define the function to train the model on one epoch."""
# pylint: disable=import-error, no-name-in-module
import torch
import tqdm

from sklearn.metrics import f1_score


def train_one_epoch(
    model, loader, f_loss, optimizer, device
):  # pylint: disable=too-many-locals
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
        f1_score (float) : f1 score over one epoch
    """

    model.train()

    n_samples = 0
    tot_loss, correct = 0.0, 0.0
    predicted_targets_all, targets_all = None, None

    for inputs, targets in tqdm.tqdm(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Compute the forward pass through the network up to the loss
        outputs = model(inputs)
        loss = f_loss(outputs, targets)

        n_samples += inputs.shape[0]
        tot_loss += inputs.shape[0] * f_loss(outputs, targets).item()
        predicted_targets = outputs.argmax(dim=1)
        correct += (predicted_targets == targets).sum().item()

        # Concat the result in order to compute f1-score
        if predicted_targets_all is None:
            predicted_targets_all = predicted_targets
            targets_all = targets
        else:
            predicted_targets_all = torch.cat(
                (predicted_targets_all, predicted_targets)
            )
            targets_all = torch.cat((targets_all, targets))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    f1_score_ = f1_score(
        y_true=targets_all.cpu().int().numpy(),
        y_pred=predicted_targets_all.cpu().int().numpy(),
        average="macro",
    )

    return tot_loss / n_samples, correct / n_samples, f1_score_
