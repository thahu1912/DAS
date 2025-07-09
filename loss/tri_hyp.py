import torch
import torch.nn as nn
import torch.nn.functional as F
import hyptorch.pmath as pmath
import hyptorch.nn as hnn
import numpy as np

import sampling
from loss._share import bl_triplets_weight

ALLOWED_MINING_OPS = list(sampling.BATCHMINING_METHODS.keys())
REQUIRES_SAMPLING = True
REQUIRES_OPTIM = False


class Criterion(nn.Module):
    """
    Hyperbolic Triplet Loss with Graph-Aware Anchor Mining
    
    1 Loss: Hyperbolic triplet loss
    1 Geometry: Poincare ball geometry
    1 Innovation: Graph-aware anchor mining/sampling
    
    Key Features:
    - Pure hyperbolic triplet loss (no regularizers)
    - Graph-aware anchor selection for better triplets
    - Learnable curvature parameter
    - Clean, focused design
    """

    def __init__(self, opt, sampling):
        super(Criterion, self).__init__()
        self.opt = opt
        self.margin = opt.loss_triplet_margin
        self.sampling = sampling
        self.name = 'tri_hyp'
        
        # Hyperbolic parameters
        self.c = getattr(opt, 'hyperbolic_c', 1.0)  # Curvature parameter
        
        # Graph-aware mining parameters
        self.k_neighbors = getattr(opt, 'k_neighbors', 5)  # Number of neighbors for graph construction
        self.temperature = getattr(opt, 'temperature', 0.1)  # Temperature for softmax
        self.label_weight = getattr(opt, 'label_weight', 0.5)  # Weight for label-aware edges
        
        # Adaptive curvature learning
        self.train_c = getattr(opt, 'train_c', False)
        if self.train_c:
            self.c = nn.Parameter(torch.tensor(self.c))
        
        # Hyperbolic distance layer
        self.hyp_distance = hnn.HyperbolicDistanceLayer(c=self.c)
        
        self.ALLOWED_MINING_OPS = ALLOWED_MINING_OPS
        self.REQUIRES_SAMPLING = REQUIRES_SAMPLING
        self.REQUIRES_OPTIM = REQUIRES_OPTIM

    def hyperbolic_triplet_distance(self, anchor, positive, negative):
        """
        Compute hyperbolic triplet distance using Poincare ball geometry
        
        Args:
            anchor: Anchor embedding
            positive: Positive embedding
            negative: Negative embedding
            
        Returns:
            Triplet loss value
        """
        # Compute hyperbolic distances
        dist_pos = self.hyp_distance(anchor, positive, c=self.c)
        dist_neg = self.hyp_distance(anchor, negative, c=self.c)
        
        # Standard triplet loss with hyperbolic distances
        triplet_loss = F.relu(dist_pos - dist_neg + self.margin)
        return triplet_loss

    def graph_aware_anchor_mining(self, embeddings, labels):
        """
        Graph-aware anchor mining: select anchors based on graph structure
        
        Innovation: Use graph connectivity to identify good anchor points
        
        Args:
            embeddings: Input embeddings
            labels: Ground truth labels
            
        Returns:
            Selected anchor indices
        """
        batch_size = embeddings.size(0)
        labels_tensor = torch.tensor(labels, device=embeddings.device)
        
        # Compute pairwise hyperbolic distances
        dist_matrix = pmath.dist_matrix(embeddings, embeddings, c=self.c)
        
        # Create label matrix
        label_matrix = (labels_tensor.unsqueeze(1) == labels_tensor.unsqueeze(0)).float()
        
        # Find k-nearest neighbors for each point
        _, indices = torch.topk(dist_matrix, k=self.k_neighbors + 1, dim=1, largest=False)
        indices = indices[:, 1:]  # Remove self
        
        # Compute graph connectivity scores
        connectivity_scores = torch.zeros(batch_size, device=embeddings.device)
        
        for i in range(batch_size):
            neighbors = indices[i]
            
            # Compute neighbor weights based on distance and labels
            neighbor_dists = dist_matrix[i, neighbors]
            neighbor_labels = labels_tensor[neighbors]
            same_label = (neighbor_labels == labels_tensor[i]).float()
            
            # Graph connectivity: high score for points with many same-class neighbors
            connectivity_scores[i] = torch.mean(same_label)
        
        # Select anchors based on connectivity scores
        # Higher connectivity = better anchor (more representative of its class)
        num_anchors = min(batch_size // 2, 32)  # Select reasonable number of anchors
        _, anchor_indices = torch.topk(connectivity_scores, k=num_anchors, largest=True)
        
        return anchor_indices

    def graph_aware_sampling(self, embeddings, labels):
        """
        Graph-aware sampling: use graph structure to guide triplet selection
        
        Innovation: Leverage graph connectivity for better triplet mining
        
        Args:
            embeddings: Input embeddings
            labels: Ground truth labels
            
        Returns:
            List of triplets (anchor, positive, negative)
        """
        batch_size = embeddings.size(0)
        labels_tensor = torch.tensor(labels, device=embeddings.device)
        
        # Get graph-aware anchors
        anchor_indices = self.graph_aware_anchor_mining(embeddings, labels)
        
        # Compute pairwise hyperbolic distances
        dist_matrix = pmath.dist_matrix(embeddings, embeddings, c=self.c)
        
        triplets = []
        
        for anchor_idx in anchor_indices:
            anchor_label = labels_tensor[anchor_idx]
            
            # Find positive samples (same class)
            positive_mask = (labels_tensor == anchor_label) & (torch.arange(batch_size, device=embeddings.device) != anchor_idx)
            positive_indices = torch.where(positive_mask)[0]
            
            # Find negative samples (different class)
            negative_mask = (labels_tensor != anchor_label)
            negative_indices = torch.where(negative_mask)[0]
            
            if len(positive_indices) > 0 and len(negative_indices) > 0:
                # Select positive based on graph connectivity
                positive_dists = dist_matrix[anchor_idx, positive_indices]
                positive_weights = F.softmax(-positive_dists / self.temperature, dim=0)
                positive_idx = positive_indices[torch.multinomial(positive_weights, 1)]
                
                # Select negative based on graph connectivity (harder negatives)
                negative_dists = dist_matrix[anchor_idx, negative_indices]
                negative_weights = F.softmax(negative_dists / self.temperature, dim=0)  # Higher weight for closer negatives
                negative_idx = negative_indices[torch.multinomial(negative_weights, 1)]
                
                triplets.append([anchor_idx.item(), positive_idx.item(), negative_idx.item()])
        
        return triplets

    def forward(self, batch, labels, **kwargs):
        """
        Forward pass for Hyperbolic Triplet Loss with Graph-Aware Mining
        
        Args:
            batch: Input embeddings
            labels: Ground truth labels
            **kwargs: Additional arguments
            
        Returns:
            Triplet loss value (no regularizers)
        """
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        # Ensure embeddings are in hyperbolic space (Poincare ball)
        batch = pmath.project(batch, c=self.c)

        # Use graph-aware sampling instead of standard sampling
        sampled_triplets = self.graph_aware_sampling(batch, labels)
        
        # If no triplets found, fall back to standard sampling
        if len(sampled_triplets) == 0:
            sampled_triplets = self.sampling(batch, labels)

        # Compute hyperbolic triplet loss
        triplet_losses = torch.stack([
            self.hyperbolic_triplet_distance(batch[triplet[0]], batch[triplet[1]], batch[triplet[2]])
            for triplet in sampled_triplets
        ])
        
        # Return pure triplet loss (no regularizers)
        triplet_loss = torch.mean(triplet_losses)

        # Logging for tensorboard
        tensorboard = kwargs.get('tensorboard', None)
        if tensorboard:
            tensorboard.add_scalar('TriHyp/TripletLoss', triplet_loss.item(), self.opt.iteration)
            tensorboard.add_scalar('TriHyp/NumTriplets', len(sampled_triplets), self.opt.iteration)
            
            # Log embedding statistics
            avg_norm = torch.mean(torch.norm(batch, dim=1)).item()
            max_norm = torch.max(torch.norm(batch, dim=1)).item()
            avg_dist_origin = torch.mean(pmath.dist0(batch, c=self.c)).item()
            
            tensorboard.add_scalar('TriHyp/AvgEmbeddingNorm', avg_norm, self.opt.iteration)
            tensorboard.add_scalar('TriHyp/MaxEmbeddingNorm', max_norm, self.opt.iteration)
            tensorboard.add_scalar('TriHyp/AvgDistFromOrigin', avg_dist_origin, self.opt.iteration)
            
            # Log curvature if trainable
            if self.train_c:
                tensorboard.add_scalar('TriHyp/Curvature', self.c.item(), self.opt.iteration)
            else:
                tensorboard.add_scalar('TriHyp/Curvature', self.c, self.opt.iteration)

        return triplet_loss

    def extra_repr(self):
        """String representation for debugging"""
        return f"TriHyp(margin={self.margin}, c={self.c}, train_c={self.train_c}, k_neighbors={self.k_neighbors})" 