import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Injects information about the relative or absolute position of the tokens in the sequence.
    This is crucial for Time-Series because without it, the Transformer doesn't know the order of candles!
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return x

class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_features, d_model=64, nhead=4, num_layers=3, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        
        self.model_type = 'Transformer'
        
        # 1. Linear layer to project our raw features (like RSI, MACD, Price) into the embedding dimension
        self.feature_extractor = nn.Linear(num_features, d_model)
        
        # 2. Positional Encoding (gives the model a sense of time/order)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. The Core Transformer Encoder Layer
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 4. Final output layer: outputs 3 probabilities (Long, Short, Flat)
        self.decoder = nn.Linear(d_model, 3)

    def forward(self, src):
        # src shape expected: (batch_size, sequence_length, num_features)
        
        # Transformers expect shape (sequence_length, batch_size, features)
        src = src.permute(1, 0, 2)
        
        # Project features to d_model space
        src = self.feature_extractor(src)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Pass through the Transformer
        output = self.transformer_encoder(src)
        
        # We only care about the last output in the sequence (the most recent candle's prediction based on the whole history)
        last_output = output[-1, :, :]
        
        # Decode to our 3 classes (Long, Short, Flat)
        logits = self.decoder(last_output)
        return logits

if __name__ == "__main__":
    # Quick Test to prove the architecture works
    # Let's say we have 10 features (Open, High, Low, Close, Volume, RSI, etc.)
    # And we look at the last 100 candles.
    batch_size = 16
    seq_length = 100
    features = 10
    
    # Create fake dummy data representing a batch of 16 charts, 100 candles each
    dummy_data = torch.randn(batch_size, seq_length, features)
    
    # Initialize the Apex Oracle Transformer
    model = TimeSeriesTransformer(num_features=features)
    
    # Run the fake data through the brain
    predictions = model(dummy_data)
    
    print(f"Input Shape: {dummy_data.shape} -> (Batch, Candles, Features)")
    print(f"Output Shape: {predictions.shape} -> (Batch, Classes [Long, Short, Flat])")
    print("\nTransformer architecture initialized successfully. Ready to learn.")
