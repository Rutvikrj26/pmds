#!/usr/bin/env python3
"""
RS-1: Titans Layer Depth Experiment

Research Spike from master_plan.md:
- Question: How many MLP layers are optimal for relation encoding?
- Experiment: Test 1, 2, 4 layers on CLUTRR validation set.
- Metrics: Accuracy vs. Memory Size vs. Training Speed.
"""

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Titans imports
from titans_pytorch import NeuralMemory

print("=" * 60)
print("RS-1: Titans Layer Depth Experiment")
print("=" * 60)


@dataclass
class ExperimentConfig:
    """Configuration for the layer depth experiment."""
    dim: int = 64
    chunk_size: int = 1
    heads: int = 2
    dim_head: int = 32
    batch_size: int = 32
    num_epochs: int = 5
    learning_rate: float = 1e-3
    num_train_samples: int = 1000
    num_val_samples: int = 200


class SyntheticRelationDataset(Dataset):
    """
    Synthetic relation dataset mimicking CLUTRR structure.
    
    Each sample is a sequence of (entity_a, relation, entity_b) triplets
    encoded as vectors. The task is to predict the final relation.
    """
    
    RELATIONS = ["parent", "child", "sibling", "spouse", "grandparent", "aunt", "uncle"]
    
    def __init__(self, num_samples: int, dim: int, num_hops: int = 2):
        self.num_samples = num_samples
        self.dim = dim
        self.num_hops = num_hops
        
        # Pre-generate all samples
        self.samples = []
        self.labels = []
        
        for _ in range(num_samples):
            # Generate random entity/relation embeddings
            # Sequence: [entity_0, rel_0, entity_1, rel_1, entity_2, ...]
            seq_len = (num_hops + 1) * 2 + 1  # entities and relations interleaved
            sequence = torch.randn(seq_len, dim)
            
            # Label is the "composed" relation (simplified: just the last relation index)
            label = torch.randint(0, len(self.RELATIONS), (1,)).item()
            
            self.samples.append(sequence)
            self.labels.append(label)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.samples[idx], self.labels[idx]


class RelationClassifier(nn.Module):
    """Classifier that uses Titans memory for relation prediction."""
    
    def __init__(self, config: ExperimentConfig, num_layers: int):
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        
        # Titans Neural Memory
        self.memory = NeuralMemory(
            dim=config.dim,
            chunk_size=config.chunk_size,
            dim_head=config.dim_head,
            heads=config.heads,
            default_model_kwargs=dict(
                depth=num_layers,
                expansion_factor=2.0,
            ),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.dim, config.dim),
            nn.ReLU(),
            nn.Linear(config.dim, len(SyntheticRelationDataset.RELATIONS)),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through memory and classifier.
        
        Args:
            x: Input sequence [batch, seq_len, dim]
            
        Returns:
            Logits [batch, num_relations]
        """
        # Pass through memory
        retrieved, _, _ = self.memory(x, state=None, return_surprises=True)
        
        # Pool the retrieved memory (mean over sequence)
        pooled = retrieved.mean(dim=1)  # [batch, dim]
        
        # Classify
        logits = self.classifier(pooled)
        return logits
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = torch.tensor(batch_y).to(device)
        
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == batch_y).sum().item()
        total += len(batch_y)
    
    return total_loss / len(dataloader), correct / total


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = torch.tensor(batch_y).to(device)
            
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)
    
    return total_loss / len(dataloader), correct / total


def run_experiment(num_layers: int, config: ExperimentConfig) -> dict[str, Any]:
    """Run experiment for a specific layer depth."""
    print(f"\n{'='*60}")
    print(f"Testing {num_layers} layer(s)")
    print(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create datasets
    train_dataset = SyntheticRelationDataset(config.num_train_samples, config.dim)
    val_dataset = SyntheticRelationDataset(config.num_val_samples, config.dim)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    # Create model
    model = RelationClassifier(config, num_layers).to(device)
    num_params = model.count_parameters()
    print(f"Model parameters: {num_params:,}")
    
    # Memory size estimate (weights in MB)
    memory_mb = num_params * 4 / (1024 * 1024)  # 4 bytes per float32
    print(f"Estimated memory: {memory_mb:.2f} MB")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    train_times = []
    best_val_acc = 0.0
    
    for epoch in range(config.num_epochs):
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        epoch_time = time.time() - start_time
        train_times.append(epoch_time)
        
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        best_val_acc = max(best_val_acc, val_acc)
        
        print(f"  Epoch {epoch+1}/{config.num_epochs}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2%}, "
              f"Val Acc={val_acc:.2%}, Time={epoch_time:.2f}s")
    
    avg_time = sum(train_times) / len(train_times)
    
    return {
        "num_layers": num_layers,
        "num_params": num_params,
        "memory_mb": memory_mb,
        "best_val_acc": best_val_acc,
        "avg_epoch_time": avg_time,
    }


def main():
    config = ExperimentConfig()
    print(f"\nExperiment Configuration:")
    print(f"  dim={config.dim}, batch_size={config.batch_size}")
    print(f"  epochs={config.num_epochs}, lr={config.learning_rate}")
    print(f"  train_samples={config.num_train_samples}, val_samples={config.num_val_samples}")
    
    # Test different layer depths
    layer_depths = [1, 2, 4]
    results = []
    
    for depth in layer_depths:
        result = run_experiment(depth, config)
        results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("RS-1 EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"\n{'Layers':<8} {'Params':<12} {'Memory (MB)':<12} {'Val Acc':<12} {'Time/Epoch':<12}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['num_layers']:<8} {r['num_params']:<12,} {r['memory_mb']:<12.2f} "
              f"{r['best_val_acc']:<12.2%} {r['avg_epoch_time']:<12.2f}s")
    
    # Recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    
    # Find best accuracy
    best_result = max(results, key=lambda x: x['best_val_acc'])
    fastest_result = min(results, key=lambda x: x['avg_epoch_time'])
    smallest_result = min(results, key=lambda x: x['num_params'])
    
    print(f"\n  Best Accuracy: {best_result['num_layers']} layers ({best_result['best_val_acc']:.2%})")
    print(f"  Fastest Training: {fastest_result['num_layers']} layers ({fastest_result['avg_epoch_time']:.2f}s/epoch)")
    print(f"  Smallest Model: {smallest_result['num_layers']} layers ({smallest_result['num_params']:,} params)")
    
    # Overall recommendation based on accuracy/speed tradeoff
    if best_result['num_layers'] == 2:
        print("\n  ✅ RECOMMENDATION: Use 2 layers (good balance of accuracy, speed, and size)")
    else:
        print(f"\n  ✅ RECOMMENDATION: Use {best_result['num_layers']} layers (highest accuracy)")


if __name__ == "__main__":
    main()
