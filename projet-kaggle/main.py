import torch
from data_processing import load_and_preprocess_data
from train import train_model
from predict import make_predictions
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, test = load_and_preprocess_data("train.csv", "test.csv", "processed_train.csv", "processed_test.csv")

    X_train, y_train = train.drop("class", axis=1).values, train["class"].values
    X_test = test.values

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)

    config = {
        "learning_rate": 0.0006,
        "epochs": 1,
        "hidden_dim": 448,
        "weight_decay": 0.0002,
    }

    model = train_model(train_loader, X_train.shape[1], config["hidden_dim"], device, config)

    test_ids = pd.read_csv("test.csv")["id"]
    make_predictions(model, X_test_tensor, test_ids, "submission.csv")

if __name__ == "__main__":
    main()
