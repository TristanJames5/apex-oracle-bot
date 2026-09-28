import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import glob
from transformer_model import TimeSeriesTransformer

class TradingDataset(Dataset):
    def __init__(self, csv_files, seq_length=60, future_lookahead=1, threshold=0.001):
        self.seq_length = seq_length
        self.samples = []
        self.labels = []
        
        print("Loading and chunking datasets for the Oracle...")
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                # Drop unnecessary columns like date strings or index columns
                feature_cols = [col for col in df.columns if col not in ['Date', 'Datetime', 'Unnamed: 0', 'index']]
                
                data = df[feature_cols].values
                
                # Check if we have 'Close' to calculate our target
                if 'Close' not in feature_cols:
                    print(f"Skipping {file}, no Close column.")
                    continue
                    
                close_idx = feature_cols.index('Close')
                
                # Create sliding windows of length `seq_length`
                for i in range(len(data) - seq_length - future_lookahead):
                    x = data[i : i + seq_length]
                    
                    current_close = data[i + seq_length - 1, close_idx]
                    future_close = data[i + seq_length - 1 + future_lookahead, close_idx]
                    
                    diff = future_close - current_close
                    
                    # Label 0: Flat, 1: Long (Up), 2: Short (Down)
                    if diff > threshold:
                        y = 1
                    elif diff < -threshold:
                        y = 2
                    else:
                        y = 0
                        
                    self.samples.append(x)
                    self.labels.append(y)
            except Exception as e:
                print(f"Failed to load {file}: {e}")
                
        self.samples = np.array(self.samples, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        print(f"Dataset created! Total sliding windows: {len(self.samples)}")
        self.num_features = self.samples.shape[2]
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx]), torch.tensor(self.labels[idx])


def train():
    csv_files = glob.glob("*_PROCESSED.csv")
    if not csv_files:
        print("No PROCESSED.csv files found.")
        return
        
    # Create dataset (60 candles looking back to predict the next 1 candle)
    dataset = TradingDataset(csv_files, seq_length=60)
    
    # We use a large batch size for fast GPU/CPU processing
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Initialize the Apex Oracle Transformer architecture
    model = TimeSeriesTransformer(num_features=dataset.num_features).to(device)
    
    # CrossEntropyLoss because this is a classification problem (Long, Short, Flat)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 5
    print("\n=============================================")
    print("IGNITING PHASE 3 ORACLE TRAINING SEQUENCE")
    print("=============================================\n")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X)
            
            # Loss and Backward pass
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(dataloader)}] | Loss: {loss.item():.4f} | Accuracy: {100 * correct / total:.2f}%")
                
        print(f">>>> Epoch {epoch+1} Summary: Avg Loss: {total_loss/len(dataloader):.4f} | Total Accuracy: {100 * correct / total:.2f}% <<<<\n")
        
    # Save the brain
    torch.save(model.state_dict(), "oracle_phase3.pth")
    print("Training Complete. Model saved as 'oracle_phase3.pth'")


if __name__ == "__main__":
    train()
