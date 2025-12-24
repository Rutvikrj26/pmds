"""
CLUTRR Dataset for Titans Pre-training.

Parses CLUTRR CSV data into sequences of relational facts for training
the Titans Memory module.
"""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from scallop_titans.constants import CLUTRR_RELATIONS


@dataclass
class ClutrrSample:
    """A single CLUTRR training sample."""
    id: str
    story: str
    query: tuple[str, str]
    target: str
    fact_chain: list[tuple[str, str, str]]  # (entity_a, relation, entity_b)
    
class ClutrrSequenceDataset(Dataset):
    """
    Dataset that parses CLUTRR CSVs into relation sequences.
    
    Extracts the 'proof_state' column which contains the logical chain
    of facts leading to the target relation.
    
    Returns:
        Dictionary containing:
        - entity_indices: LongTensor [seq_len]
        - relation_indices: LongTensor [seq_len]
        - query_entity_pair: LongTensor [2]
        - target_relation_id: LongTensor [1]
    """
    
    # CLUTRR relations (imported from shared constants)
    RELATION_TYPES = CLUTRR_RELATIONS
    
    def __init__(self, csv_path: str | Path, num_entities: int = 10000):
        self.csv_path = Path(csv_path)
        self.num_entities = num_entities
        self.relation_to_id = {r: i for i, r in enumerate(self.RELATION_TYPES)}
        
        self.samples: list[ClutrrSample] = []
        self._load_data()
        
    def _load_data(self):
        """Parse the CSV file."""
        import pandas as pd
        df = pd.read_csv(self.csv_path)
        
        for _, row in df.iterrows():
            try:
                # Parse proof_state string roughly to extract facts
                # Format: [{('Head', 'rel', 'Tail'): [('A', 'r1', 'B'), ('B', 'r2', 'C')]}]
                proof_state_str = row['proof_state']
                
                # Careful evaluation of the literal string representation
                proof_data = ast.literal_eval(proof_state_str)
                
                # Extract the fact chain from the first proof
                # proof_data is a list of dicts. We take first dict.
                if not proof_data:
                    continue
                    
                first_proof = proof_data[0]
                # Keys are target triples, values are list of supporting facts
                # We want the supporting facts sequence
                fact_chain = []
                for _, facts in first_proof.items():
                    fact_chain = facts
                    break
                
                # Parse query tuple string "('A', 'B')"
                query_tuple = ast.literal_eval(row['query'])
                
                self.samples.append(ClutrrSample(
                    id=str(row['id']),
                    story=row['story'],
                    query=query_tuple,
                    target=row['target_text'],
                    fact_chain=fact_chain
                ))
            except (ValueError, SyntaxError, KeyError) as e:
                # Skip malformed rows
                continue
                
    def __len__(self) -> int:
        return len(self.samples)
        
    def _hash_entity(self, name: str) -> int:
        """Deterministic hash of entity name to index."""
        import zlib
        # CRC32 is fast and deterministic
        return zlib.crc32(name.encode()) % self.num_entities
        
    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        
        # 1. Convert fact chain to indices
        entity_indices = []
        relation_indices = []
        
        # Flatten chain: A r1 B, B r2 C -> A, r1, B, B, r2, C
        # For Titans, we need inputs. Let's format as (Head, Rel, Tail) triplets
        # sequences.
        
        # We will return the raw sequences for the collator to pad
        seq_heads = []
        seq_rels = []
        seq_tails = []
        
        for head, rel, tail in sample.fact_chain:
            # Map relation to ID, defaulting to unknown if not found (shouldn't happen in clean data)
            if rel not in self.relation_to_id:
                continue
                
            seq_heads.append(self._hash_entity(head))
            seq_rels.append(self.relation_to_id[rel])
            seq_tails.append(self._hash_entity(tail))
            
        # 2. Convert Query
        query_head = self._hash_entity(sample.query[0])
        query_tail = self._hash_entity(sample.query[1])
        
        # 3. Target
        if sample.target not in self.relation_to_id:
            # Fallback or skip - for now mapped to -1 to ignore in loss
            target_id = -1
        else:
            target_id = self.relation_to_id[sample.target]
            
        return {
            "seq_heads": torch.tensor(seq_heads, dtype=torch.long),
            "seq_rels": torch.tensor(seq_rels, dtype=torch.long),
            "seq_tails": torch.tensor(seq_tails, dtype=torch.long),
            "query_head": torch.tensor(query_head, dtype=torch.long),
            "query_tail": torch.tensor(query_tail, dtype=torch.long),
            "target_id": torch.tensor(target_id, dtype=torch.long),
        }
        
    def get_samples(self) -> list[ClutrrSample]:
        """Expose raw samples for trace conversion."""
        return self.samples

def collate_clutrr(batch: list[dict[str, Any]]) -> dict[str, Tensor]:
    """Collate batch with padding for sequences."""
    from torch.nn.utils.rnn import pad_sequence
    
    # Pad sequences
    seq_heads = pad_sequence([b["seq_heads"] for b in batch], batch_first=True, padding_value=0)
    seq_rels = pad_sequence([b["seq_rels"] for b in batch], batch_first=True, padding_value=0)
    seq_tails = pad_sequence([b["seq_tails"] for b in batch], batch_first=True, padding_value=0)
    
    # Stack fixed-size items
    query_heads = torch.stack([b["query_head"] for b in batch])
    query_tails = torch.stack([b["query_tail"] for b in batch])
    target_ids = torch.stack([b["target_id"] for b in batch])
    
    # Create attention mask (1 for real data, 0 for pad)
    # Using seq_heads length as reference
    mask = (seq_heads != 0) # Keep as BoolTensor
    
    return {
        "seq_heads": seq_heads,
        "seq_rels": seq_rels,
        "seq_tails": seq_tails,
        "mask": mask,
        "query_head": query_heads,
        "query_tail": query_tails,
        "target_id": target_ids
    }
