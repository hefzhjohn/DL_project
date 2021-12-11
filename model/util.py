# train the model
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader


def train_model(
    data, model, epoch: int = 10, batch_size: int = 1024, debug: bool = False
):

    # Split data into batches
    data = DataLoader(data, batch_size=batch_size, shuffle=True)

    # define the optimization
    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    # enumerate epochs
    for ep in range(epoch):

        # enumerate mini batches
        for i, (inputs, targets) in enumerate(data):

            # clear the gradients
            optimizer.zero_grad()

            # compute the model output
            yhat = model(inputs.float())

            # calculate loss
            loss = criterion(yhat, targets.float())

            # credit assignment
            loss.backward()

            # update model weights
            optimizer.step()

            if debug and i % 1000 == 0:
                print(f"Epoch {ep}, batch {i}, loss: {loss.data.item()}")
