# train.py

import torch
from sklearn.metrics import f1_score, accuracy_score
from torch_geometric.loader import NeighborLoader
import torch.nn as nn
class Trainer:
    def __init__(self, model, data, device='cpu', epochs=20, batch_size=128, lr=0.005):
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_f1': [], 'val_f1': []}
        
        # Create masks for training and validation
        torch.manual_seed(42)
        num_nodes = data['tweet'].num_nodes
        perm = torch.randperm(num_nodes)
        train_idx = perm[:int(0.8 * num_nodes)]
        val_idx = perm[int(0.8 * num_nodes):int(0.9 * num_nodes)]
        test_idx = perm[int(0.9 * num_nodes):]
        data['tweet'].train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data['tweet'].val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data['tweet'].test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data['tweet'].train_mask[train_idx] = True
        data['tweet'].val_mask[val_idx] = True
        data['tweet'].test_mask[test_idx] = True
    
    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(self.data.x_dict, self.data.edge_index_dict)
            mask = self.data['tweet'].train_mask
            loss = self.criterion(out[mask], self.data['tweet'].y[mask])
            loss.backward()
            self.optimizer.step()
            
            # Training metrics
            train_loss = loss.item()
            train_acc, train_f1 = self.evaluate(split='train')
            val_loss, val_acc, val_f1 = self.evaluate(split='val')
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    def evaluate(self, split='val'):
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x_dict, self.data.edge_index_dict)
            if split == 'train':
                mask = self.data['tweet'].train_mask
            elif split == 'val':
                mask = self.data['tweet'].val_mask
            elif split == 'test':
                mask = self.data['tweet'].test_mask
            else:
                raise ValueError("Invalid split")
            loss = self.criterion(out[mask], self.data['tweet'].y[mask]).item()
            pred = out[mask].argmax(dim=1)
            y_true = self.data['tweet'].y[mask].cpu().numpy()
            y_pred = pred.cpu().numpy()
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted')
        return loss, acc, f1
