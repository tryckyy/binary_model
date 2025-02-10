import torch.optim as optim
import torch.nn as nn
from model import NeuralNetwork
from utils import calculate_accuracy
import wandb

def train_model(train_loader, input_dim, hidden_dim, device, config):
    wandb.init(
        project="projet-kaggle",
        name = "run",
        config=config
    )
    model = NeuralNetwork(input_dim, hidden_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    for epoch in range(config["epochs"]):
        model.train()
        running_loss, running_accuracy = 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            accuracy = calculate_accuracy(outputs, labels)
            running_loss += loss.item()
            running_accuracy += accuracy

        avg_loss = running_loss / len(train_loader)
        avg_accuracy = running_accuracy / len(train_loader)

        wandb.log({
            "Epoch Loss": avg_loss,
            "Epoch Accuracy": avg_accuracy,
            "Learning Rate": optimizer.param_groups[0]['lr'],
        })
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {avg_loss}, Accuracy: {avg_accuracy}")

    return model