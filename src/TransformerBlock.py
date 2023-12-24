import torch.nn as nn
# Define a basic transformer block

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
        )
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention(x, x, x)[0]
        x = x + self.dropout(attended)
        x = self.norm1(x)
        fedforward = self.feedforward(x)
        x = x + self.dropout(fedforward)
        x = self.norm2(x)
        return x
