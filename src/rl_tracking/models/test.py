import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from torchmetrics.classification import Accuracy

class SimpleNN(pl.LightningModule):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 128)  # Input layer to hidden layer
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)# Hidden layer to output layer
        self.fc4 = nn.Linear(128, 3)
        self.criterion = nn.CrossEntropyLoss()  # Loss function for classification
        self.train_acc = Accuracy(num_classes=3, task='multiclass')
        self.val_acc = Accuracy(num_classes=3, task='multiclass')

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def training_step(self, batch, batch_idx):
        # Unpack data
        data, labels = batch
        # Forward pass
        outputs = self(data)
        # Compute loss
        loss = self.criterion(outputs, labels)
        # Return the loss
        acc = self.train_acc(outputs, labels)
        self.log('train_acc', acc, prog_bar=True)

        self.log('train_loss', loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)

        # Calculate accuracy
        acc = self.val_acc(outputs, y)

        # Log validation accuracy and loss
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        # Use Adam optimizer
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def on_epoch_end(self):
        # Access the loss for the epoch
        loss = self.trainer.callback_metrics.get('train_loss')
        if loss is not None:
            print(f"Epoch {self.current_epoch} - Loss: {loss:.4f}")