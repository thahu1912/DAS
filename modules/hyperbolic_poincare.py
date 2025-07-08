import random
import torch
import torch.nn.functional as F
import math


class HyperbolicPoincare:
    """
    Hyperbolic Poincaré ball model operations for enhancing densely anchored sampling.
    
    The Poincaré ball model represents hyperbolic space as the interior of a unit ball
    in Euclidean space, where distances grow exponentially as we move toward the boundary.
    This is particularly useful for hierarchical data structures and can improve
    the quality of embeddings by better preserving hierarchical relationships.
    """
    
    def __init__(self, dim, curvature=-1.0, eps=1e-8):
        """
        Initialize hyperbolic Poincaré operations.
        
        Args:
            dim: Dimension of the embedding space
            curvature: Negative curvature of hyperbolic space (default: -1.0)
            eps: Small epsilon for numerical stability
        """
        self.dim = dim
        self.curvature = curvature
        self.eps = eps
        
    def to_poincare(self, x, eps=None):
        """
        Project Euclidean embeddings to Poincaré ball.
        
        Args:
            x: Euclidean embeddings (N, dim)
            eps: Numerical stability constant
            
        Returns:
            Poincaré embeddings (N, dim)
        """
        if eps is None:
            eps = self.eps
            
        # Ensure embeddings are within the unit ball
        norm = torch.norm(x, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=eps, max=1.0 - eps)
        
        # Project to Poincaré ball
        poincare_x = x / (1.0 + torch.sqrt(1.0 - norm**2))
        return poincare_x
    
    def to_euclidean(self, x, eps=None):
        """
        Project Poincaré embeddings back to Euclidean space.
        
        Args:
            x: Poincaré embeddings (N, dim)
            eps: Numerical stability constant
            
        Returns:
            Euclidean embeddings (N, dim)
        """
        if eps is None:
            eps = self.eps
            
        norm = torch.norm(x, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=eps, max=1.0 - eps)
        
        # Project back to Euclidean space
        euclidean_x = 2.0 * x / (1.0 - norm**2)
        return euclidean_x
    
    def poincare_distance(self, x, y, eps=None):
        """
        Compute Poincaré distance between two points.
        
        Args:
            x: First point in Poincaré ball (N, dim)
            y: Second point in Poincaré ball (N, dim)
            eps: Numerical stability constant
            
        Returns:
            Poincaré distances (N,)
        """
        if eps is None:
            eps = self.eps
            
        diff = x - y
        norm_diff = torch.norm(diff, dim=-1)
        norm_x = torch.norm(x, dim=-1)
        norm_y = torch.norm(y, dim=-1)
        
        # Poincaré distance formula
        numerator = 2.0 * norm_diff**2
        denominator = (1.0 - norm_x**2) * (1.0 - norm_y**2)
        denominator = torch.clamp(denominator, min=eps)
        
        distance = torch.acosh(1.0 + numerator / denominator)
        return distance
    
    def poincare_norm(self, x, eps=None):
        """
        Compute Poincaré norm of a point.
        
        Args:
            x: Point in Poincaré ball (N, dim)
            eps: Numerical stability constant
            
        Returns:
            Poincaré norms (N,)
        """
        if eps is None:
            eps = self.eps
            
        norm = torch.norm(x, dim=-1)
        norm = torch.clamp(norm, min=eps, max=1.0 - eps)
        
        poincare_norm = 2.0 * torch.atanh(norm)
        return poincare_norm
    
    def poincare_add(self, x, y, eps=None):
        """
        Poincaré addition (Möbius addition).
        
        Args:
            x: First point in Poincaré ball (N, dim)
            y: Second point in Poincaré ball (N, dim)
            eps: Numerical stability constant
            
        Returns:
            Result of Poincaré addition (N, dim)
        """
        if eps is None:
            eps = self.eps
            
        norm_x = torch.norm(x, dim=-1, keepdim=True)
        norm_y = torch.norm(y, dim=-1, keepdim=True)
        norm_x = torch.clamp(norm_x, min=eps, max=1.0 - eps)
        norm_y = torch.clamp(norm_y, min=eps, max=1.0 - eps)
        
        # Möbius addition formula
        dot_product = torch.sum(x * y, dim=-1, keepdim=True)
        
        numerator = (1.0 + 2.0 * dot_product + norm_y**2) * x + (1.0 - norm_x**2) * y
        denominator = 1.0 + 2.0 * dot_product + norm_x**2 * norm_y**2
        denominator = torch.clamp(denominator, min=eps)
        
        result = numerator / denominator
        return result
    
    def poincare_scalar_mul(self, r, x, eps=None):
        """
        Poincaré scalar multiplication.
        
        Args:
            r: Scalar (N,) or (N, 1)
            x: Point in Poincaré ball (N, dim)
            eps: Numerical stability constant
            
        Returns:
            Result of Poincaré scalar multiplication (N, dim)
        """
        if eps is None:
            eps = self.eps
            
        norm = torch.norm(x, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=eps, max=1.0 - eps)
        
        # Poincaré scalar multiplication formula
        tanh_r = torch.tanh(r.unsqueeze(-1) * torch.atanh(norm))
        result = tanh_r * x / norm
        return result
    
    def poincare_exp_map(self, v, x, eps=None):
        """
        Exponential map in Poincaré ball.
        
        Args:
            v: Tangent vector at x (N, dim)
            x: Base point in Poincaré ball (N, dim)
            eps: Numerical stability constant
            
        Returns:
            Result of exponential map (N, dim)
        """
        if eps is None:
            eps = self.eps
            
        norm_v = torch.norm(v, dim=-1, keepdim=True)
        norm_x = torch.norm(x, dim=-1, keepdim=True)
        norm_x = torch.clamp(norm_x, min=eps, max=1.0 - eps)
        
        # Exponential map formula
        lambda_x = 2.0 / (1.0 - norm_x**2)
        lambda_x = torch.clamp(lambda_x, min=eps)
        
        tanh_term = torch.tanh(lambda_x * norm_v / 2.0)
        result = self.poincare_add(x, tanh_term * v / norm_v)
        return result
    
    def poincare_log_map(self, y, x, eps=None):
        """
        Logarithmic map in Poincaré ball.
        
        Args:
            y: Target point in Poincaré ball (N, dim)
            x: Base point in Poincaré ball (N, dim)
            eps: Numerical stability constant
            
        Returns:
            Tangent vector (N, dim)
        """
        if eps is None:
            eps = self.eps
            
        diff = self.poincare_add(-x, y)  # Poincaré subtraction
        norm_diff = torch.norm(diff, dim=-1, keepdim=True)
        norm_x = torch.norm(x, dim=-1, keepdim=True)
        norm_x = torch.clamp(norm_x, min=eps, max=1.0 - eps)
        
        # Logarithmic map formula
        lambda_x = 2.0 / (1.0 - norm_x**2)
        lambda_x = torch.clamp(lambda_x, min=eps)
        
        result = 2.0 * torch.atanh(norm_diff) * diff / (lambda_x * norm_diff)
        return result
    
    def hyperbolic_sampling(self, embeddings, targets, num_samples=3, temperature=1.0, eps=None):
        """
        Perform hyperbolic-aware sampling in Poincaré ball.
        
        Args:
            embeddings: Input embeddings (N, dim)
            targets: Target labels (N,)
            num_samples: Number of samples to generate
            temperature: Sampling temperature
            eps: Numerical stability constant
            
        Returns:
            Sampled embeddings (num_samples * N, dim), labels (num_samples * N,)
        """
        if eps is None:
            eps = self.eps
            
        device = embeddings.device
        N, dim = embeddings.shape
        
        # Convert to Poincaré ball
        poincare_embeddings = self.to_poincare(embeddings, eps)
        
        sampled_embeddings = []
        sampled_labels = []
        
        for _ in range(num_samples):
            # Generate random tangent vectors
            tangent_vectors = torch.randn(N, dim, device=device) * temperature
            
            # Apply exponential map to generate new points
            new_embeddings = self.poincare_exp_map(tangent_vectors, poincare_embeddings, eps)
            
            # Convert back to Euclidean space
            euclidean_embeddings = self.to_euclidean(new_embeddings, eps)
            
            sampled_embeddings.append(euclidean_embeddings)
            sampled_labels.append(targets.clone())
        
        # Concatenate all samples
        all_embeddings = torch.cat(sampled_embeddings, dim=0)
        all_labels = torch.cat(sampled_labels, dim=0)
        
        return all_embeddings, all_labels
    
    def hyperbolic_interpolation(self, x, y, t, eps=None):
        """
        Hyperbolic interpolation between two points.
        
        Args:
            x: First point in Poincaré ball (N, dim)
            y: Second point in Poincaré ball (N, dim)
            t: Interpolation parameter (N,) or scalar
            eps: Numerical stability constant
            
        Returns:
            Interpolated point (N, dim)
        """
        if eps is None:
            eps = self.eps
            
        # Convert to Poincaré ball if needed
        if torch.norm(x, dim=-1).max() > 1.0 - eps:
            x = self.to_poincare(x, eps)
        if torch.norm(y, dim=-1).max() > 1.0 - eps:
            y = self.to_poincare(y, eps)
        
        # Compute geodesic interpolation
        log_map = self.poincare_log_map(y, x, eps)
        interpolated = self.poincare_exp_map(t.unsqueeze(-1) * log_map, x, eps)
        
        return interpolated
    
    def hyperbolic_centroid(self, points, weights=None, eps=None, max_iter=100):
        """
        Compute hyperbolic centroid using iterative algorithm.
        
        Args:
            points: Points in Poincaré ball (N, dim)
            weights: Weights for each point (N,) or None for uniform
            eps: Numerical stability constant
            max_iter: Maximum iterations for convergence
            
        Returns:
            Hyperbolic centroid (dim,)
        """
        if eps is None:
            eps = self.eps
            
        if weights is None:
            weights = torch.ones(points.shape[0], device=points.device)
        
        # Initialize centroid as weighted average
        centroid = torch.sum(weights.unsqueeze(-1) * points, dim=0) / torch.sum(weights)
        centroid = self.to_poincare(centroid.unsqueeze(0), eps).squeeze(0)
        
        for _ in range(max_iter):
            # Compute weighted sum of log maps
            log_maps = []
            for i, point in enumerate(points):
                log_map = self.poincare_log_map(point.unsqueeze(0), centroid.unsqueeze(0), eps)
                log_maps.append(weights[i] * log_map.squeeze(0))
            
            # Update centroid
            total_log_map = torch.stack(log_maps).sum(dim=0)
            centroid = self.poincare_exp_map(total_log_map.unsqueeze(0), centroid.unsqueeze(0), eps).squeeze(0)
            
            # Check convergence
            if torch.norm(total_log_map) < eps:
                break
        
        return centroid


class HyperbolicDenselyAnchoredSampling:
    """
    Enhanced Densely Anchored Sampling with Hyperbolic Poincaré geometry.
    
    This class extends the original DAS with hyperbolic operations to better
    preserve hierarchical relationships and improve embedding quality.
    """
    
    def __init__(self, num_classes, dim, 
                 num_produce=3, normalize=True,
                 dfs_num_scale=4, dfs_scale_range=(0.5, 2.0),
                 mts_num_transformation_bank=10, mts_scale=0.01,
                 hyperbolic_weight=0.5, curvature=-1.0, detach=True):
        """
        Initialize Hyperbolic DAS.
        
        Args:
            num_classes: Number of classes
            dim: Embedding dimension
            num_produce: Number of samples to produce
            normalize: Whether to normalize embeddings
            dfs_num_scale: Number of top features for DFS
            dfs_scale_range: Range for DFS scaling
            mts_num_transformation_bank: Size of transformation bank
            mts_scale: Scale for MTS transformations
            hyperbolic_weight: Weight for hyperbolic operations (0-1)
            curvature: Hyperbolic curvature
            detach: Whether to detach gradients
        """
        self.num_classes = num_classes
        self.dim = dim
        self.num_produce = num_produce
        self.normalize = normalize
        self.dfs_num_scale = dfs_num_scale
        self.dfs_scale_range = dfs_scale_range
        self.mts_num_transformation_bank = mts_num_transformation_bank
        self.mts_scale = mts_scale
        self.hyperbolic_weight = hyperbolic_weight
        self.detach = detach
        
        # Initialize hyperbolic operations
        self.hyperbolic = HyperbolicPoincare(dim, curvature)
        
        # Frequency recorder matrix for DFS
        self.frequency_recorder_matrix = {k: [0 for _ in range(dim)] for k in range(num_classes)}
        
        # Hyperbolic transformation banks
        self.hyperbolic_banks = [
            HyperbolicMemoryBank(queue_size=mts_num_transformation_bank, queue_dim=dim, hyperbolic=self.hyperbolic) 
            for _ in range(num_classes)
        ]
    
    def __call__(self, embeddings, targets):
        """
        Generate enhanced embeddings using hyperbolic DAS.
        
        Args:
            embeddings: Input embeddings (N, dim)
            targets: Target labels (N,)
            
        Returns:
            Produced embeddings list, produced targets list
        """
        # Update frequency recorder matrix
        self._update_frequency_recorder_matrix(embeddings, targets)
        
        # Update hyperbolic transformation banks
        self._update_hyperbolic_banks(embeddings, targets)
        
        # Produce enhanced embeddings
        produced_embeddings, produced_targets = self._produce_hyperbolic(embeddings, targets)
        
        return produced_embeddings, produced_targets
    
    @torch.no_grad()
    def _update_frequency_recorder_matrix(self, embeddings, targets):
        """Update frequency recorder matrix for DFS."""
        for embedding, target in zip(embeddings, targets):
            _, topK_index = torch.topk(embedding, k=self.dfs_num_scale, dim=0, largest=True)
            for ind in topK_index:
                self.frequency_recorder_matrix[target.item()][ind.item()] += 1
    
    def _update_hyperbolic_banks(self, embeddings, targets):
        """Update hyperbolic transformation banks."""
        detached_embeddings = embeddings.clone().detach()
        unique_targets = torch.unique(targets)
        
        for t in unique_targets:
            self.hyperbolic_banks[t].detach()
            
            # Get embeddings for this class
            target_embeddings = detached_embeddings[targets == t]
            
            if target_embeddings.size(0) > 1:
                # Convert to Poincaré ball
                poincare_embeddings = self.hyperbolic.to_poincare(target_embeddings)
                
                # Compute hyperbolic transformations
                transformations = []
                for i in range(poincare_embeddings.size(0)):
                    for j in range(i + 1, poincare_embeddings.size(0)):
                        # Compute hyperbolic difference
                        diff = self.hyperbolic.poincare_log_map(
                            poincare_embeddings[j:j+1], 
                            poincare_embeddings[i:i+1]
                        ).squeeze(0)
                        transformations.append(diff)
                
                if transformations:
                    transformations = torch.stack(transformations)
                    self.hyperbolic_banks[t].enqueue_dequeue(
                        transformations, 
                        torch.zeros(transformations.size(0), dtype=torch.long) + t
                    )
    
    def _produce_hyperbolic(self, embeddings, targets):
        """Produce embeddings using hyperbolic operations."""
        device = embeddings.device
        produced_embeddings, produced_targets = [], []
        
        for _ in range(self.num_produce):
            # Get topK indices for DFS
            topK_index = []
            for target in targets:
                _, _topK_index = torch.topk(
                    torch.Tensor(self.frequency_recorder_matrix[target.item()]),
                    k=self.dfs_num_scale, dim=0, largest=True
                )
                topK_index.append(_topK_index)
            topK_index = torch.stack(topK_index, dim=0).to(device)
            
            # Generate scaling factors
            scale = (self.dfs_scale_range[1] - self.dfs_scale_range[0]) * torch.rand(
                embeddings.size(0), self.dfs_num_scale, device=device
            ) + self.dfs_scale_range[0]
            
            # Generate hyperbolic transformations
            hyperbolic_transforms = []
            for _topK_index, target in zip(topK_index, targets):
                transformations, _ = self.hyperbolic_banks[target].get()
                
                if transformations.size(0) > 1:
                    rand_index = random.randint(0, transformations.size(0) - 1)
                else:
                    rand_index = 0
                
                if transformations.size(0) == 0:
                    _transform = torch.zeros(embeddings.size(-1), device=device)
                else:
                    _transform = transformations[rand_index, :]
                hyperbolic_transforms.append(_transform)
            
            hyperbolic_transforms = torch.stack(hyperbolic_transforms, dim=0)
            hyperbolic_transforms = self.mts_scale * hyperbolic_transforms
            
            # Create enhanced embeddings
            if not self.detach:
                _produced_embeddings = embeddings.clone()
            else:
                _produced_embeddings = embeddings.clone().detach()
            
            # Apply DFS scaling
            _part_generated_embeddings = _produced_embeddings.gather(dim=1, index=topK_index)
            _part_generated_embeddings = _part_generated_embeddings * scale
            _produced_embeddings = _produced_embeddings.scatter(
                dim=1, index=topK_index, src=_part_generated_embeddings
            )
            
            # Apply hyperbolic transformations
            poincare_embeddings = self.hyperbolic.to_poincare(_produced_embeddings)
            poincare_transforms = self.hyperbolic.to_poincare(hyperbolic_transforms)
            
            # Combine Euclidean and hyperbolic operations
            euclidean_result = _produced_embeddings + hyperbolic_transforms
            hyperbolic_result = self.hyperbolic.poincare_exp_map(
                poincare_transforms, poincare_embeddings
            )
            hyperbolic_result = self.hyperbolic.to_euclidean(hyperbolic_result)
            
            # Weighted combination
            final_embeddings = (1 - self.hyperbolic_weight) * euclidean_result + \
                              self.hyperbolic_weight * hyperbolic_result
            
            # Normalize if required
            if self.normalize:
                final_embeddings = F.normalize(final_embeddings, dim=1)
            
            produced_embeddings.append(final_embeddings)
            produced_targets.append(targets.clone())
        
        return produced_embeddings, produced_targets


class HyperbolicMemoryBank:
    """
    Memory bank for storing hyperbolic transformations.
    """
    
    def __init__(self, queue_size, queue_dim, hyperbolic):
        self.queue_size = queue_size
        self.queue_dim = queue_dim
        self.hyperbolic = hyperbolic
        self.ptr = 0
        self.transformations = torch.zeros(queue_size, queue_dim).cuda()
        self.targets = torch.zeros(queue_size, dtype=torch.long)
    
    @property
    def is_full(self):
        return self.targets[-1].item() != 0
    
    @property
    def size(self):
        if self.is_full:
            return self.queue_size
        else:
            return self.ptr
    
    def get(self):
        filter_index = torch.sum(torch.abs(self.transformations), dim=1) > 0
        return self.transformations[filter_index], self.targets[filter_index]
    
    def enqueue_dequeue(self, trans, targets):
        q_size = len(targets)
        
        if q_size > self.queue_size:
            self.transformations = trans[-self.queue_size:]
            self.targets = targets[-self.queue_size:]
        elif self.ptr + q_size > self.queue_size:
            end_ptr = min(self.ptr + q_size, self.queue_size)
            remain = q_size - (end_ptr - self.ptr)
            
            self.transformations[self.ptr:end_ptr] = trans[:-remain]
            self.targets[self.ptr:end_ptr] = targets[:-remain]
            
            self.transformations[:remain] = trans[-remain:]
            self.targets[:remain] = targets[-remain:]
            
            self.ptr = remain
        else:
            self.transformations[self.ptr:self.ptr + q_size] = trans
            self.targets[self.ptr:self.ptr + q_size] = targets
            self.ptr += q_size
    
    def detach(self):
        self.transformations = self.transformations.detach() 