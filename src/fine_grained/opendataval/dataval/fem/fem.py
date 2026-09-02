import torch
import numpy as np
from opendataval.dataval.progress import ProgressBar, progress_range
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader, TensorDataset
from opendataval.dataval.api import DataEvaluator, ModelMixin

class ForgettingEvents(DataEvaluator, ModelMixin):
    def __init__(self, epochs=50, batch_size=64, random_state=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = check_random_state(random_state)

    def input_data(self, x_train, y_train, x_valid, y_valid):
        # Convert one-hot labels to class indices if needed
        if isinstance(y_train, np.ndarray) and y_train.ndim > 1:
            y_train = np.argmax(y_train, axis=1)

        # Convert to torch tensors if not already
        if not isinstance(x_train, torch.Tensor):
            x_train = torch.tensor(x_train, dtype=torch.float32)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train, dtype=torch.long)

        self.x_train = x_train
        self.y_train = y_train
        self.num_points = len(x_train)
        return self

    def train_data_values(self, *args, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.pred_model.clone().to(device)

        if isinstance(self.x_train, torch.Tensor):
            dataset = TensorDataset(self.x_train, self.y_train)
            full_x = self.x_train.to(device)
            ground_truth = self.y_train
        else:
            dataset = self.x_train  # assumed Subset or custom dataset
            full_x = torch.stack([dataset[i][0] for i in range(len(dataset))]).to(device)
            ground_truth = torch.tensor([dataset[i][1] for i in range(len(dataset))])

        if ground_truth.ndim > 1:
            ground_truth = torch.argmax(ground_truth, dim=1)

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        correct_history = torch.zeros((self.num_points, self.epochs), dtype=torch.bool)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        for epoch in progress_range(self.epochs, "Tracking forgetting events"):
            model.train()
            for batch in loader:
                print(f"Batch type: {type(batch)}")
                print(f"Batch content type: {type(batch[0]) if isinstance(batch, (tuple, list)) else 'N/A'}")
                print(f"Batch length: {len(batch) if hasattr(batch, '__len__') else 'N/A'}")
                break
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).long()

                optimizer.zero_grad()
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

            # Evaluate on full training set
            model.eval()
            with torch.no_grad():
                logits = model(full_x)
                preds = torch.argmax(logits, dim=1)
                correct = preds.cpu() == ground_truth.cpu()
                correct_history[:, epoch] = correct

        # Count forgetting events
        forgetting_counts = np.zeros(self.num_points, dtype=int)
        ever_learned = np.zeros(self.num_points, dtype=bool)

        for i in range(self.num_points):
            for t in range(1, self.epochs):
                if correct_history[i, t-1] and not correct_history[i, t]:
                    forgetting_counts[i] += 1
                if correct_history[i, t]:
                    ever_learned[i] = True

        max_forgetting = np.max(forgetting_counts)
        values = 1 - (forgetting_counts / (max_forgetting + 1e-8))
        values[~ever_learned] = 0.0

        self.data_values = values
        return self

    def evaluate_data_values(self):
        return self.data_values
