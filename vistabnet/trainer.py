from focal_loss.focal_loss import FocalLoss
from .vistabnet import TabularVisionTransformer
import numpy as np
import torch
import tqdm


def balanced_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    classes = np.unique(y_true)
    accs = []
    for c in classes:
        idx = y_true == c
        acc = (y_true[idx] == y_pred[idx]).sum() / idx.sum()
        accs.append(acc)
    return np.mean(accs)

class VisTabNetClassifier:
    def __init__(self, input_features, classes, projections=195, projection_depth=4, lr=1e-4, proj_lr=3e-4, device="cuda:0"):
        self.device = device
        self.device_type = torch.device(device).type

        self.input_features = input_features
        self.classes = classes
        self.projections = projections
        self.projection_depth = projection_depth

        model = TabularVisionTransformer(self.input_features, self.classes, self.projections, self.projection_depth,
            image_size=224,
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072
        ).to(torch.float16).to(self.device)

        self.model = model

        # Prepare optimizer
        params = list(filter(lambda p: p.requires_grad, self.model.head.parameters()))
        params_projector = list(filter(lambda p: p.requires_grad, self.model.projector.parameters()))
        self.optim = torch.optim.Adam([
        {
            'params': params,
            'lr': lr,
        },
        {
            'params': params_projector,
            'lr': proj_lr,
        }
        ], eps=1e-4)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lambda epoch: 0.98 ** epoch)
        self.criterion = FocalLoss(gamma=1.5)

    def _prepare_dataloader(self, X, y, batch_size, shuffle=False):
        X = torch.tensor(X)
        y = torch.tensor(y)
    
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader

    def fit(self, X, y, eval_X=None, eval_y=None, lr=1e-4, proj_lr=3e-4, epochs=100, batch_size=32, verbose=True):
        NOISE_STD = 0.01

        def training_step(model, optim, criterion, X, y):
            X = X.to(self.device).to(torch.float16)
            y = y.to(self.device)

            X = X + torch.randn(X.shape).to(self.device) * NOISE_STD

            optim.zero_grad()
            with torch.autocast(device_type=self.device_type, dtype=torch.float16, enabled=True):
                y_pred_ = model(X)

            assert torch.all(torch.isfinite(y_pred_))
            assert torch.all(y_pred_ >= 0)
            assert torch.all(y_pred_ <= 1)
            loss = criterion(y_pred_, y)
            loss.backward()
            optim.step()
            return y,y_pred_, loss.item()

        # Prepare dataloaders
        trainloader = self._prepare_dataloader(X, y, batch_size=batch_size, shuffle=True)
        testloader = None
        if eval_X is not None and eval_y is not None:
            testloader = self._prepare_dataloader(eval_X, eval_y, batch_size=batch_size, shuffle=False)

        # Train
        with tqdm.trange(epochs, disable=not verbose) as t:
            for epoch in t:
                self.model.train()
                epoch_loss = 0
                y_true = []
                y_pred = []
                for X, y in trainloader:
                    y, y_pred_, ls = training_step(self.model, self.optim, self.criterion, X, y)
                    epoch_loss += ls
                    
                    y_true.extend(y.cpu().tolist())
                    y_pred.extend(y_pred_.argmax(dim=1).cpu().tolist())

                train_bacc = balanced_accuracy(y_true, y_pred)
                self.scheduler.step()
                self.model.eval()

                if testloader is not None:
                    y_true = []
                    y_pred = []
                    with torch.inference_mode():
                        for X, y in testloader:
                            X = X.to(self.device).to(torch.float16)
                            y = y.to(self.device)
                            with torch.autocast(device_type=self.device_type, dtype=torch.float16, enabled=True):
                                y_pred_ = self.model(X)

                            y_true.extend(y.cpu().tolist())
                            y_pred.extend(y_pred_.argmax(dim=1).cpu().tolist())

                    bacc = balanced_accuracy(y_true, y_pred)
                    t.set_description(f"Epoch {epoch} loss: {epoch_loss:.4f} train_bacc: {train_bacc:.4f} eval_bacc: {bacc:.4f}")
                else:
                    t.set_description(f"Epoch {epoch} loss: {epoch_loss:.4f} train_bacc: {train_bacc:.4f}")
                    
            
    def predict(self, X):
        batch_size = 32
        self.model.eval()
        with torch.inference_mode():
            with torch.autocast(device_type=self.device_type, dtype=torch.float16, enabled=True):
                y_pred = []
                for i in range(0, len(X), batch_size):
                    X_ = X[i:i+batch_size]
                    X_ = torch.tensor(X_).to(self.device).to(torch.float16)
                    y_pred_ = self.model(X_)
                    y_pred.append(y_pred_.cpu())
                y_pred = torch.cat(y_pred, dim=0)
            y_pred = y_pred.argmax(dim=1).tolist()
        return y_pred

    
    def predict_proba(self, X):
        batch_size = 32

        self.model.eval()
        with torch.inference_mode():
            with torch.autocast(device_type=self.device_type, dtype=torch.float16, enabled=True):
                y_pred = []
                for i in range(0, len(X), batch_size):
                    X_ = X[i:i+batch_size]
                    X_ = torch.tensor(X_).to(self.device).to(torch.float16)
                    y_pred_ = self.model(X_).cpu()
                    y_pred.append(y_pred_)
                y_pred = torch.cat(y_pred, dim=0)
        return y_pred
  