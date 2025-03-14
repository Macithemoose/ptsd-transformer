import torch
import torch.nn as nn

class MLP_Attn_Block(nn.Module):
    def __init__(self, d_model = 40, time_in = 500, time_out = 25, n_heads = 1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads)
        self.dense = nn.Linear(d_model, d_model)
        self.reduce = nn.Linear(time_in, time_out)

    def forward(self, x):
        x = self.dense(x)
        # MultiheadAttention expects (time, batch, embed_dim)
        x_transposed = x.permute(1, 0, 2)  # => (500, batch_size, 40)
        attn_output, attn_weights = self.mha(x_transposed, x_transposed, x_transposed)
        # Convert back to (batch_size, 500, 40)
        attn_output = attn_output.permute(1, 0, 2)

        # Want: (batch_size, 40, 500)
        attn_output = attn_output.permute(0, 2, 1)

        # Another dense layer to reduce to (batch, 40, 25):
        reduced = self.reduce(attn_output)

        final_output = reduced.permute(0, 2, 1)  # (batch_size, 25, 40)

        return final_output

class EmpathyTransformer(nn.Module):
    def __init__(self, d_model=1000, n_heads=4, num_transformer_layers=1):
        super().__init__()
        # Create 20 blocks for the 20 segments
        self.num_segments = 20
        self.blocks = nn.ModuleList([MLP_Attn_Block(d_model=40, n_heads=1) for _ in range(self.num_segments)])

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        self.voter = nn.Linear(d_model, 1)
    
    def forward(self, x):
        batch_size = x.shape[0]

        # Split into segments:
        segment_length = 500
        segments = []
        for i in range(self.num_segments):
            start = i * segment_length
            end = start + segment_length
            segment = x[:, start:end, :]  # (batch_size, 500, 40)
            segments.append(segment)

        # Send each segment through the dense + attention block defined above
        block_outputs = []
        for i, block in enumerate(self.blocks):
            out = block(segments[i])  # (batch_size, 25, 40)
            # Flatten to (batch_size, 25*40) = (batch_size, 1000)
            out_flat = out.reshape(batch_size, -1)  # (batch_size, 1000)
            block_outputs.append(out_flat)
        
        # Stack flattened segments along sequence dimension
        x_stacked = torch.stack(block_outputs, dim=1)

        # Pass through Transformer Encoder layer
        transformer_output = self.transformer_encoder(x_stacked)

        # We take the mean across the 20 sequences
        pooled = transformer_output.mean(dim=1) # => (batch_size, 1000)

        # Voter layer
        logits = self.voter(pooled)

        return logits
   