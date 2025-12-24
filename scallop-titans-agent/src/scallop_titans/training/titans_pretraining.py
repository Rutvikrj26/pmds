"""
Titans Pre-training Script.

Trains the Titans Memory module on CLUTRR relational sequences using
surprise-based learning.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from scallop_titans.data.clutrr_dataset import ClutrrSequenceDataset, collate_clutrr
from scallop_titans.memory.titans_adapter import MemoryConfig, TitansMemoryAdapter

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

@dataclass
class TrainerConfig:
    data_path: Path
    save_dir: Path
    batch_size: int = 64
    lr: float = 1e-4
    epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4

class TitansPretrainer:
    def __init__(self, model: TitansMemoryAdapter, config: TrainerConfig):
        self.model = model
        self.config = config
        self.model.to(config.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.lr,
            weight_decay=0.01
        )
        
    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.config.device) for k, v in batch.items()}
            
            self.optimizer.zero_grad()
            
            # Forward pass
            logits, surprises = self.model.forward_sequence(
                seq_heads=batch["seq_heads"],
                seq_rels=batch["seq_rels"],
                seq_tails=batch["seq_tails"],
                mask=batch["mask"],
                query_head=batch["query_head"],
                query_tail=batch["query_tail"]
            )
            
            # Compute Loss
            # Target ID -1 means skip (unknown relation)
            target = batch["target_id"]
            valid_mask = target != -1
            
            if not valid_mask.any():
                continue
                
            loss = F.cross_entropy(logits[valid_mask], target[valid_mask])
            
            # Add surprise regularization (optional, encourages memory usage)
            # Higher surprise = more learning. 
            # We assume the model handles surprise internally for weight updates,
            # but we can add a term here if needed. For now, just primary task loss.
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Metrics
            preds = torch.argmax(logits, dim=-1)
            correct += (preds[valid_mask] == target[valid_mask]).sum().item()
            total += valid_mask.sum().item()
            total_loss += loss.item()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.2%}"})
            
        return {
            "loss": total_loss / len(dataloader),
            "accuracy": correct / total if total > 0 else 0.0
        }
        
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.eval()
        correct = 0
        total = 0
        
        for batch in dataloader:
            batch = {k: v.to(self.config.device) for k, v in batch.items()}
            
            logits, _ = self.model.forward_sequence(
                seq_heads=batch["seq_heads"],
                seq_rels=batch["seq_rels"],
                seq_tails=batch["seq_tails"],
                mask=batch["mask"],
                query_head=batch["query_head"],
                query_tail=batch["query_tail"]
            )
            
            target = batch["target_id"]
            valid_mask = target != -1
            
            preds = torch.argmax(logits, dim=-1)
            correct += (preds[valid_mask] == target[valid_mask]).sum().item()
            total += valid_mask.sum().item()
            
        return {"accuracy": correct / total if total > 0 else 0.0}
        
    def save_checkpoint(self, path: Path, metrics: dict):
        torch.save({
            "model_state": self.model.state_dict(),
            "config": asdict(self.model.config),
            "metrics": metrics
        }, path)
        logger.info(f"Saved checkpoint to {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to CLUTRR csv")
    parser.add_argument("--save-dir", type=str, default="checkpoints/titans")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    
    # Setup
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Data
    dataset = ClutrrSequenceDataset(args.data)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    logger.info(f"Loaded {len(dataset)} samples. Train: {train_size}, Val: {val_size}")
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=64, 
        shuffle=True, 
        collate_fn=collate_clutrr,
        num_workers=4
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=64, 
        collate_fn=collate_clutrr
    )
    
    # Model
    mem_config = MemoryConfig(dim=256, num_layers=2)
    model = TitansMemoryAdapter(mem_config)
    
    # Trainer
    config = TrainerConfig(
        data_path=Path(args.data), 
        save_dir=save_dir,
        epochs=args.epochs
    )
    trainer = TitansPretrainer(model, config)
    
    # Train Loop
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        logger.info(f"Epoch {epoch}/{args.epochs}")
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader)
        
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2%}")
        logger.info(f"Val Acc: {val_metrics['accuracy']:.2%}")
        
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            trainer.save_checkpoint(save_dir / "best_model.pt", val_metrics)
            
if __name__ == "__main__":
    main()
