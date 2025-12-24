"""
Unit tests for CLUTRR Dataset.
"""
import pytest
import torch
from pathlib import Path
from tempfile import NamedTemporaryFile
import csv
import pandas as pd

from scallop_titans.data.clutrr_dataset import ClutrrSequenceDataset, collate_clutrr

@pytest.fixture
def mock_clutrr_csv():
    """Create a mock CLUTRR csv file for testing."""
    data = [
        {
            "id": "uuid1", 
            "story": "Alice is Bob's mother. Bob is Carol's father.", 
            "query": "('Alice', 'Carol')", 
            "target_text": "grandmother",
            "proof_state": "[{('Alice', 'grandmother', 'Carol'): [('Alice', 'mother', 'Bob'), ('Bob', 'father', 'Carol')]}]"
        },
        {
            "id": "uuid2", 
            "story": "Dave is Eve's brother.", 
            "query": "('Dave', 'Eve')", 
            "target_text": "brother",
            "proof_state": "[{('Dave', 'brother', 'Eve'): [('Dave', 'brother', 'Eve')]}]"
        }
    ]
    
    with NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        path = Path(f.name)
        
    yield path
    # Cleanup
    if path.exists():
        path.unlink()

def test_dataset_loading(mock_clutrr_csv):
    """Test loading data from CSV."""
    ds = ClutrrSequenceDataset(mock_clutrr_csv)
    assert len(ds) == 2
    
def test_sample_structure(mock_clutrr_csv):
    """Test parsed sample content."""
    ds = ClutrrSequenceDataset(mock_clutrr_csv)
    
    # Check first sample (grandmother chain)
    sample = ds[0]
    
    # Chain: Alice->Bob, Bob->Carol (2 steps)
    assert len(sample["seq_heads"]) == 2
    assert len(sample["seq_rels"]) == 2
    
    # Relations: mother(6), father(5) based on fixed list order
    # mother is idx 6, father is idx 5 in RELATION_TYPES
    assert sample["seq_rels"][0] == 6 
    assert sample["seq_rels"][1] == 5
    
    # Target: grandmother (idx 7)
    assert sample["target_id"] == 7
    
def test_hashing_consistency(mock_clutrr_csv):
    """Test that entity hashing is deterministic."""
    ds = ClutrrSequenceDataset(mock_clutrr_csv)
    
    # Hash "Alice" manually
    import zlib
    alice_hash = zlib.crc32(b"Alice") % 10000
    
    # Get metadata from first sample
    # Alice is head of first fact
    assert ds[0]["seq_heads"][0] == alice_hash
    
    # Alice is query head
    assert ds[0]["query_head"] == alice_hash

def test_collate_fn(mock_clutrr_csv):
    """Test batch collation and padding."""
    ds = ClutrrSequenceDataset(mock_clutrr_csv)
    batch = [ds[0], ds[1]]
    
    collated = collate_clutrr(batch)
    
    # seq_heads should be padded to max length (2)
    assert collated["seq_heads"].shape == (2, 2)
    
    # First sample has 2 facts, Second has 1 fact
    # Mask should reflect this
    assert torch.all(collated["mask"][0] == 1)  # [1, 1]
    assert collated["mask"][1][0] == 1
    assert collated["mask"][1][1] == 0          # Padded
