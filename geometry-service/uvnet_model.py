"""
UV-Net Model Architecture
Minimal implementation for loading pre-trained checkpoint
Based on References/UV-Net-main/uvnet/models.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import dgl
from dgl.nn.pytorch import GraphConv, EdgeConv

class UVNetEncoder(nn.Module):
    """UV-Net Graph Encoder"""
    def __init__(self, in_channels=7, hidden_dim=128, num_layers=3):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_channels, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GraphConv(hidden_dim, hidden_dim))
        
        self.edge_conv = EdgeConv(hidden_dim, hidden_dim, batch_norm=True)
    
    def forward(self, g, node_feat, edge_feat):
        h = node_feat
        
        for conv in self.convs:
            h = F.relu(conv(g, h))
        
        # Edge convolution
        if edge_feat is not None and edge_feat.shape[0] > 0:
            h = self.edge_conv(g, h)
        
        return h


class Segmentation(pl.LightningModule):
    """
    UV-Net Segmentation Model
    Simplified version for inference only
    """
    def __init__(self, crv_in_channels=6, num_classes=16, hidden_dim=128):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Encoder
        self.encoder = UVNetEncoder(
            in_channels=7,  # 3D points + normals + mask
            hidden_dim=hidden_dim,
            num_layers=3
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, g):
        """
        Forward pass for inference
        
        Args:
            g: DGL graph with node features in g.ndata['x']
        
        Returns:
            logits: [num_nodes, num_classes]
        """
        node_feat = g.ndata["x"]
        edge_feat = g.edata.get("x", None)
        
        # Encode
        h = self.encoder(g, node_feat, edge_feat)
        
        # Classify
        logits = self.classifier(h)
        
        return logits
    
    def predict(self, g):
        """Convenience method for inference"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(g)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return preds, probs
