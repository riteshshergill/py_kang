import torch
import torch.nn.functional as F
import math
from tqdm import tqdm
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class LinearLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        dropout_prob=0.2,
        activation=nn.ReLU
    ):
        super(LinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = activation()
        self.batch_norm = nn.BatchNorm1d(out_features)

    def forward(self, x):
        """
        Perform a forward pass through the LinearLayer.

        Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_features) after applying
                      linear transformation, batch normalization, activation function,
                      and dropout.
        """
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class LinearL(nn.Module):
    def __init__(
        self,
        layers_hidden,
        dropout_prob=0.5
    ):
        super(LinearL, self).__init__()

        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                LinearLayer(
                    in_features,
                    out_features,
                    dropout_prob=dropout_prob
                )
            )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def train_model(self, model, trainloader, valloader, optimizer, scheduler, criterion, device, epochs):
        """
        Train the model.

        Args:
        trainloader (DataLoader): DataLoader for training data.
        valloader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the training on (CPU or GPU).
        epochs (int): Number of epochs to train the model.

        Returns:
        None
        """
        model.to(device)
        best_val_loss = float('inf')
        early_stop_counter = 0
        early_stop_patience = 7
        for epoch in range(epochs):
            model.train()
            with tqdm(trainloader) as pbar:
                for i, (images, labels) in enumerate(pbar):
                    # Move images and labels to the specified device
                    images = images.view(-1, 28 * 28).to(device)

                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    # Forward pass
                    output = model(images)
                    # Compute loss
                    loss = criterion(output, labels.to(device))
                    # Backward pass
                    loss.backward()
                    # Clip gradients to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    # Update model parameters
                    optimizer.step()

                    accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
                    # Update progress bar
                    pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])

            model.eval()
            val_loss = 0
            val_accuracy = 0
            with torch.no_grad():
                for images, labels in valloader:
                    images = images.view(-1, 28 * 28).to(device)
                    output = model(images)
                    loss = criterion(output, labels.to(device))
                    val_loss += loss.item()
                    val_accuracy += (output.argmax(dim=1) == labels.to(device)).float().mean().item()

            # Average the validation loss and accuracy
            val_loss /= len(valloader)
            val_accuracy /= len(valloader)
            # Adjust learning rate based on validation loss
            scheduler.step(val_loss)

            print(f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    print('Early stopping triggered.')
                    break

        print('Finished Training')

    def test_model(self, model, testloader, device, num_samples=10):
        """
        Evaluate the KAN model on test data.

        Args:
        model (nn.Module): The trained model to be evaluated.
        testloader (DataLoader): DataLoader for test data.
        device (torch.device): Device to run the evaluation on (CPU or GPU).
        num_samples (int): Number of test samples to visualize and print.

        Returns:
        tuple: A tuple containing predictions, ground truths, and images to show.
               - predictions (list): List of predicted labels for the samples.
               - ground_truths (list): List of ground truth labels for the samples.
               - images_to_show (list): List of images corresponding to the samples.
        """
        model.to(device)
        model.eval()
        predictions = []
        ground_truths = []
        images_to_show = []

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for i, (images, labels) in enumerate(testloader):
                images = images.view(-1, 28 * 28).to(device)
                output = model(images)
                predictions.extend(output.argmax(dim=1).cpu().numpy())
                ground_truths.extend(labels.cpu().numpy())
                images_to_show.extend(images.view(-1, 28, 28).cpu().numpy())

                if len(predictions) >= num_samples:
                    break

        # Print the predictions for the specified number of samples
        for i in range(num_samples):
            print(f"Ground Truth: {ground_truths[i]}, Prediction: {predictions[i]}")

        return predictions[:num_samples], ground_truths[:num_samples], images_to_show[:num_samples]
