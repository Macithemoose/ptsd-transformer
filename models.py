import torch
import torch.nn as nn

class MLP_Attn_Block(nn.Module):
    def __init__(self, d_model = 40, time_in = 100, time_out = 25, n_heads = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads)
        self.dense = nn.Linear(time_in, time_out)

    def forward(self, x):
        # Transpose x:
        x = x.permute(1, 0, 2) 

        attn_output, _ = self.mha(x, x, x)

        # Transpose for linear layer
        attn_output = attn_output.permute(1, 0, 2)

        reduced = self.dense(attn_output.permute(0, 2, 1)) # Back to (batch, features, time)

        # Expects (batch_size, time, features)
        final = reduced.permute(0, 2, 1)

        return final

class EmpathyTransformer(nn.Module):
    def __init__(self, d_model=1000, n_heads=4, num_transformer_layers=1):
        super().__init__()
        # Create 5 blocks for the 5 segments
        self.num_segments = 5
        self.blocks = nn.ModuleList([MLP_Attn_Block(d_model=40, n_heads=4) for _ in range(self.num_segments)])

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        self.voter = nn.Linear(d_model, 1)
    
    def forward(self, x):
        batch_size = x.shape[0]

        # Split into segments:
        segment_length = 100
        segments = []
        for i in range(self.num_segments):
            start = i * segment_length
            end = start + segment_length
            segment = x[:, start:end, :]  # (batch_size, 100, 40)
            segments.append(segment)

        # Send each segment through the dense + attention block defined above
        block_outputs = []
        for i, block in enumerate(self.blocks):
            out = block(segments[i])  # (batch_size, 25, 40)
            # Flatten to (batch_size, 25*40) = (batch_size, 1000)
            out_flat = out.reshape(batch_size, -1)  # (batch_size, 1000)
            block_outputs.append(out_flat)
        
        # Stack flattened segments along sequence dimension *****
        x_stacked = torch.stack(block_outputs, dim=1)

        # Pass through Transformer Encoder layer
        transformer_output = self.transformer_encoder(x_stacked)

        # We take the mean across the 5 sequences
        pooled = transformer_output.mean(dim=1) # => (batch_size, 1000)

        # Voter layer
        logits = self.voter(pooled)

        return logits
   