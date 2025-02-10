import pandas as pd
import torch

def make_predictions(model, X_test_tensor, test_ids, output_file):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        y_pred_class = (torch.sigmoid(y_pred) > 0.5).float().cpu().numpy()

    submission = pd.DataFrame({
        'id': test_ids,
        'class': ["p" if pred == 1 else "e" for pred in y_pred_class]
    })
    submission.to_csv(output_file, index=False)
    print(f"Fichier de soumission sauvegardé : {output_file}")