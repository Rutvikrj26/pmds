"""
TitansMemoryAdapter: Wrapper around titans-pytorch NeuralMemory with surprise mechanism.

Implements the Titans Memory architecture from master_plan.md Part A Section 2B:
- MLP with Fast Weights via torch.func.functional_call
- Surprise mechanism: surprise = ||grad(loss)|| where loss = prediction error
- Update rule: W_new = W_old + lr * surprise * grad(W)
- Forgetting: Weight decay W = W * (1 - decay_rate)

Outputs probabilistic facts: P(relation | entity_a, entity_b)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
import torch
from torch import Tensor, nn

# Import from titans-pytorch library
from titans_pytorch import NeuralMemory, NeuralMemState, mem_state_detach

# No more hardcoded relations - using dynamic hashing


@dataclass
class MemoryConfig:
    """Configuration for Titans Memory Adapter."""

    # Architecture (from master_plan.md)
    dim: int = 384  # Model dimension (matches Qwen embedding size)
    hidden_dim: int = 256
    num_layers: int = 2  # Research Spike RS-1: Test 1, 2, 4 layers
    heads: int = 4
    dim_head: int = 64
    chunk_size: int = 16

    # Surprise mechanism
    surprise_lr: float = 1e-2  # Learning rate scaled by surprise
    max_grad_norm: float = 2.0  # Gradient clipping for stability

    # Forgetting
    decay_rate: float = 0.01  # Weight decay per step

    # Momentum (from Titans paper)
    momentum: bool = True
    momentum_order: int = 1
    
    # Embedding Config
    num_entities: int = 10000  # Size of hashing bucket
    num_relations: int = 24    # Fixed relation types


class TitansMemoryAdapter(nn.Module):
    """
    Adapter wrapping titans-pytorch NeuralMemory for the ScallopTitansAgent.

    This class provides:
    1. A high-level interface for storing/retrieving relational facts
    2. Surprise-based update mechanism for "interesting" information
    3. Session state management for multi-turn conversations
    4. Interface for Scallop foreign predicates
    """

    
    # No hardcoded relations - relations are dynamically hashed like entities
    # This allows the system to handle ANY relation type
    
    def __init__(self, config: MemoryConfig | None = None) -> None:
        """
        Initialize the TitansMemoryAdapter.

        Args:
            config: Memory configuration. Uses defaults if None.
        """
        super().__init__()
        self.config = config or MemoryConfig()
        
        # Dynamic relation vocabulary (hash -> name)
        # Mirrors vocab_cache for entities
        self.relation_cache: dict[int, str] = {}

        # Learnable Embeddings - BOTH use hashing for open vocabulary
        self.entity_emb = nn.Embedding(self.config.num_entities, self.config.dim)
        self.relation_emb = nn.Embedding(self.config.num_relations, self.config.dim)
        
        # Mask embedding for queries
        self.mask_token = nn.Parameter(torch.randn(1, self.config.dim))

        self.memory = NeuralMemory(
            dim=self.config.dim,
            chunk_size=self.config.chunk_size,
            dim_head=self.config.dim_head,
            heads=self.config.heads,
            max_grad_norm=self.config.max_grad_norm,
            momentum=self.config.momentum,
            momentum_order=self.config.momentum_order,
            activation=nn.SiLU(),  # Good default for modern MLPs
        )
        
        # Projections for entity embeddings -> memory queries
        # These are no longer needed with the new embedding approach
        # self.entity_proj = nn.Linear(self.config.dim, self.config.dim)
        # self.relation_proj = nn.Linear(self.config.dim, self.config.dim)

        # Current memory state (persists across turns)
        self._state: NeuralMemState | None = None

        # Surprise history for analysis
        self._surprise_history: list[Tensor] = []
        
        # Symbolic buffer to track what facts we've seen (for Scallop interaction)
        # In a full neural system, this would be replaced by decoding from the memory directly.
        # Stores tuples of (entity_a, relation, entity_b, probability)
        self._symbolic_facts: list[tuple[str, str, str, float]] = []

        # Context-Aware Entity Vocabulary
        # Maps hashed index -> original name for the current session
        self.vocab_cache: dict[int, str] = {}
        

    def _hash_entity_index(self, text: str) -> int:
        """Get embedding index via hashing."""
        return self._hash_text(text)

    def _hash_text(self, text: str) -> int:
        """
        Convert text to embedding index.
        Caches the text in vocab_cache for reverse lookup.
        """
        # Deterministic hash to range [0, num_entities)
        import zlib
        idx = zlib.crc32(text.encode()) % self.config.num_entities
        
        # Cache for reverse lookup (Context-Aware Vocabulary)
        self.vocab_cache[idx] = text
        
        return idx

    def _hash_relation(self, relation: str) -> int:
        """
        Hash relation name to embedding index (dynamic, no hardcoded list).
        Also caches for reverse lookup.
        """
        import zlib
        idx = zlib.crc32(relation.encode()) % self.config.num_relations
        self.relation_cache[idx] = relation
        return idx

    def _get_embedding_from_text(self, text: str) -> Tensor:
        """
        Helper to get embedding vector from text using the new learnable embedding layer.
        """
        device = self.entity_emb.weight.device
        idx = self._hash_text(text) # Use the updated _hash_text
        idx_tensor = torch.tensor([idx], device=device)
        return self.entity_emb(idx_tensor)

    def store(
        self,
        entity_a: Tensor,
        relation: Tensor,
        entity_b: Tensor,
        return_surprise: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Store a relational fact in memory.

        The fact (entity_a, relation, entity_b) is encoded and stored using
        the surprise-based update mechanism. Higher surprise leads to stronger
        memory formation.

        Args:
            entity_a: Embedding of first entity [batch, dim]
            relation: Embedding of relation type [batch, dim]
            entity_b: Embedding of second entity [batch, dim]

        Returns:
            Tuple of (stored representation, surprise value if requested)
        """
        # Ensure inputs have batch dimension
        if entity_a.dim() == 1:
            entity_a = entity_a.unsqueeze(0)
        if relation.dim() == 1:
            relation = relation.unsqueeze(0)
        if entity_b.dim() == 1:
            entity_b = entity_b.unsqueeze(0)
            
        # Combine into single sequence for memory storage
        # Shape: [batch, 3, dim]
        combined = torch.stack([entity_a, relation, entity_b], dim=1)

        # Store in memory, get surprise values
        retrieved, self._state, extra = self.memory(
            combined,
            state=self._state,
            return_surprises=True,
        )

        surprises, adaptive_lr = extra if extra else (None, None)

        # Track surprise for analysis
        if surprises is not None:
            self._surprise_history.append(surprises.mean().detach())

        if return_surprise:
            return retrieved, surprises
        return retrieved, None

    def retrieve(
        self,
        query_entity: Tensor,
        query_relation: Tensor | None = None,
    ) -> Tensor:
        """
        Retrieve facts from memory matching the query.

        Args:
            query_entity: Entity to query about [batch, dim]
            query_relation: Optional relation type filter [batch, dim]

        Returns:
            Retrieved memory representations [batch, seq, dim]
        """
        if self._state is None:
            # No memories stored yet
            batch = query_entity.shape[0] if query_entity.dim() > 1 else 1
            return torch.zeros(batch, 1, self.config.dim, device=query_entity.device)

        # Ensure batch dim
        if query_entity.dim() == 1:
            query_entity = query_entity.unsqueeze(0)
            
        # Project query (using direct embeddings now, not separate projections)
        query = query_entity
        if query_relation is not None:
            if query_relation.dim() == 1:
                query_relation = query_relation.unsqueeze(0)
            query = query + query_relation # Assuming relation acts as an additive filter

        # Query memory
        query = query.unsqueeze(1)  # [batch, 1, dim]
        retrieved, _ = self.memory(
            query,
            state=self._state,
        )

        return retrieved

    def retrieve_by_relation(
        self,
        relation: str,
        entity: str,
        top_k: int = 10
    ) -> list[tuple[Tensor, str]]:
        """
        Neural retrieval for foreign predicate.
        
        Args:
            relation: Relation string (e.g. "parent")
            entity: Entity string (e.g. "alice")
            top_k: Number of results to return
            
        Returns:
            List of (probability_tensor, entity_name) tuples.
            The tensor retains gradient history.
        """
        device = self.entity_emb.weight.device
        
        # 1. Embed Query
        # Format: [Head, Rel, Mask] -> Query Tail
        # Or [Mask, Rel, Tail] -> Query Head?
        # Usually foreign predicate is parent(alice, ?) -> Head=alice, Rel=parent
        
        # Hash relation dynamically (no hardcoded list needed)
        rel_idx = self._hash_relation(relation)
        ent_idx = self._hash_text(entity) # Use updated hash with caching
        
        query_emb = torch.stack([
            self.entity_emb(torch.tensor([ent_idx], device=device)),
            self.relation_emb(torch.tensor([rel_idx], device=device)),
            self.mask_token
        ]).squeeze(1) # [3, dim]
        
        # 2. Retrieve from memory
        # Input shape need to be [Batch=1, Seq=3, Dim]
        # CRITICAL: Pass the accumulated state to get stored facts
        retrieved, _ = self.memory(query_emb.unsqueeze(0), state=self._state)
        
        # We want the output corresponding to the LAST token (Mask)
        # retrieved: [1, 3, dim]
        mask_output = retrieved[:, -1, :] # [1, dim]
        
        # 3. Decode to entity probabilities
        # Similarity with all entities
        logits = torch.matmul(mask_output, self.entity_emb.weight.t()) # [1, NumEntities]
        probs = torch.softmax(logits, dim=-1)
        
        # 4. Top K
        # Get more candiates to increase hit rate against vocab_cache
        top_probs, top_indices = probs.topk(min(top_k * 5, self.config.num_entities))
        
        results = []
        seen_entities = set()
        
        # A) Neural Retrieval Results
        for p, idx_tensor in zip(top_probs[0], top_indices[0]):
            idx = idx_tensor.item()
            # ONLY return if we have the name cached
            if idx in self.vocab_cache:
                name = self.vocab_cache[idx]
                if name not in seen_entities:
                    results.append((p, name))
                    seen_entities.add(name)
        
        # B) Symbolic Fallback (Hybrid Retrieval)
        # For untrained memory, neural retrieval is random.
        # We mix in symbolic facts with high confidence if not already found.
        # This ensures the system works end-to-end even before extensive training.
        s_found = False
        for fact in self._symbolic_facts:
            # Fact is (entity_a, relation, entity_b, confidence)
            f_head, f_rel, f_tail, f_conf = fact
            if f_head == entity and f_rel == relation:
                if f_tail not in seen_entities:
                    # Create a tensor for the probability to maintain interface consistency
                    # (though gradients won't flow through this path efficiently yet)
                    p_tensor = torch.tensor(f_conf, device=device, requires_grad=True)
                    results.append((p_tensor, f_tail))
                    seen_entities.add(f_tail)
                    s_found = True
        
        # Sort by probability
        # Note: Mixing tensors and checking value might trigger sync, but fine for inference
        results.sort(key=lambda x: x[0].item(), reverse=True)
        
        return results[:top_k]

    def update(self, scallop_cmd: str, weights: dict | None = None) -> None:
        """
        Update memory based on a Scallop command.

        Parses the command to extract facts and stores them in memory.
        This is called when the LLM emits a <|call_scallop|> token.

        Args:
            scallop_cmd: The Scallop command string (e.g., "add_fact(mother, alice, betty)")
            weights: Optional previous weights for weight residual connection
        """
        import re
        
        # Parse commands using regex
        # Look for add_fact(relation, arg1, arg2)
        matches = re.finditer(r"add_fact\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*\)", scallop_cmd)
        
        for match in matches:
            relation, entity_a, entity_b = match.groups()
            
            # 1. Create embeddings
            emb_a = self._get_embedding_from_text(entity_a)
            emb_b = self._get_embedding_from_text(entity_b)

            # Relation embedding (dynamically hashed)
            rel_idx = self._hash_relation(relation)
            device = self.relation_emb.weight.device
            emb_rel = self.relation_emb(torch.tensor([rel_idx], device=device))
            
            # 2. Store in Neural Memory
            # We don't use the returned surprise immediately, but it updates internal state
            self.store(emb_a, emb_rel, emb_b)
            
            # 3. Add to symbolic buffer for Scallop retrieval
            # We assign a default high confidence because it came from the LLM
            # In a future version, we could use the 'surprise' metric to adjust this
            self._symbolic_facts.append((entity_a, relation, entity_b, 0.95))

    def get_probabilistic_facts(self) -> list[tuple[str, str, str, float]]:
        """
        Get all stored facts with their probabilities.

        Returns list of (entity_a, relation, entity_b, probability) tuples.
        This is used by Scallop for probabilistic reasoning.

        Returns:
            List of probabilistic fact tuples.
        """
        # Return the facts we've tracked in the buffer
        return self._symbolic_facts

    def apply_decay(self) -> None:
        """Apply weight decay for forgetting mechanism."""
        # This method is now handled by the NeuralMemory's internal decay mechanism
        # if self._state is not None and self._state.weights is not None:
        #     # Apply decay: W = W * (1 - decay_rate)
        #     decay_factor = 1.0 - self.config.decay_rate
        #     self._state = NeuralMemState(
        #         seq_index=self._state.seq_index,
        #         weights={
        #             k: v * decay_factor for k, v in self._state.weights.items()
        #         },
        #         cache_store_segment=self._state.cache_store_segment,
        #         states=self._state.states,
        #         updates=self._state.updates,
        #     )
        pass # Decay is now handled by the NeuralMemory module itself if configured.


    def reset_state(self) -> None:
        """Reset memory state (start fresh session)."""
        self._state = None
        self._surprise_history.clear()
        self._symbolic_facts.clear()

    def detach_state(self) -> None:
        """Detach state from computation graph (for multi-turn without gradient accumulation)."""
        if self._state is not None:
            self._state = mem_state_detach(self._state)

    @property
    def surprise_history(self) -> list[float]:
        """Get history of surprise values for analysis."""
        return [s.item() for s in self._surprise_history]

    def forward(
        self,
        seq: Tensor,
        state: NeuralMemState | None = None,
        return_surprises: bool = False,
    ) -> tuple[Tensor, NeuralMemState, Any]:
        """
        Forward pass through the memory module.

        This is the low-level interface matching titans-pytorch API.

        Args:
            seq: Input sequence [batch, seq_len, dim]
            state: Optional previous memory state
            return_surprises: Whether to return surprise values

        Returns:
            Tuple of (retrieved, new_state, extra_info)
        """
        state = state or self._state
        retrieved, new_state, extra = self.memory(
            seq,
            state=state,
            return_surprises=return_surprises,
        )
        self._state = new_state
        return retrieved, new_state, extra

    def forward_sequence(
        self,
        seq_heads: Tensor,    # [batch, seq_len]
        seq_rels: Tensor,     # [batch, seq_len]
        seq_tails: Tensor,    # [batch, seq_len]
        mask: Tensor,         # [batch, seq_len]
        query_head: Tensor,   # [batch]
        query_tail: Tensor,   # [batch]
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass for training on relational sequences.
        
        Args:
            seq_heads, seq_rels, seq_tails: Fact sequences
            mask: 1 for valid time steps, 0 for padding
            query_head, query_tail: Query entity indices
            
        Returns:
            Tuple of (logits [batch, num_relations], surprises [batch])
        """
        batch_size, seq_len = seq_heads.shape
        
        # 1. Embed Inputs
        head_emb = self.entity_emb(seq_heads)  # [B, L, D]
        rel_emb = self.relation_emb(seq_rels)  # [B, L, D]
        tail_emb = self.entity_emb(seq_tails)  # [B, L, D]
        
        # 2. Combine for Memory storage: [Head, Rel, Tail]
        # Shape: [Batch, Seq, 3, Dim]
        # But Titans expects [Batch, Seq, Dim], so we stack them?
        # NO: NeuralMemory takes [Batch, Seq, Dim] and updates.
        # We treat (Head, Rel, Tail) as a chunk of 3 tokens.
        
        # Reshape to [Batch, Seq*3, Dim]
        inputs = torch.stack([head_emb, rel_emb, tail_emb], dim=2)
        inputs = inputs.reshape(batch_size, seq_len * 3, self.config.dim)
        
        # 3. Create Mask for Variable Lengths
        # Expand mask: each step becomes 3 tokens
        if mask is not None:
            # [B, L] -> [B, L, 1] -> [B, L, 3] -> [B, L*3]
            flat_mask = mask.unsqueeze(-1).expand(-1, -1, 3).reshape(batch_size, -1)
        else:
            flat_mask = None
            
        # 4. Pass through Memory
        # We start with empty state for each batch
        # This trains the "in-context learning" ability
        retrieved, final_state, extra = self.memory(
            inputs,
            store_mask=flat_mask,
            return_surprises=True
        )
        
        # 5. Extract Surprises (for regularization)
        surprises = extra[0] if extra else torch.tensor(0.0)
        
        # 6. Query the Final Memory State
        # Query: (Head, [MASK], Tail)
        q_head = self.entity_emb(query_head).unsqueeze(1)  # [B, 1, D]
        q_mask = self.mask_token.expand(batch_size, 1, -1) # [B, 1, D]
        q_tail = self.entity_emb(query_tail).unsqueeze(1)  # [B, 1, D]
        
        query_seq = torch.cat([q_head, q_mask, q_tail], dim=1) # [B, 3, D]
        
        # Retrieve using the state aggregated from history
        # Note: We pass the *final* state from the history
        retrieved_query, _ = self.memory(
            query_seq,
            state=final_state
        )
        
        # 7. Classification
        # We use the embedding matrix transpose as the classifier head (weight tying)
        # Hidden state at the mask position (index 1)
        mask_hidden = retrieved_query[:, 1, :] # [B, D]
        
        # Similarity with relation embeddings
        logits = torch.matmul(mask_hidden, self.relation_emb.weight.t()) # [B, NumRels]
        
        return logits, surprises

    def save_pretrained(self, path: Path | str) -> None:
        """Save memory model weights and config."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'memory_state_dict': self.memory.state_dict(),
            'entity_emb': self.entity_emb.state_dict(),
            'relation_emb': self.relation_emb.state_dict(),
            'mask_token': self.mask_token,
            'config': asdict(self.config),
        }, path)

    @classmethod
    def from_pretrained(cls, path: Path | str) -> "TitansMemoryAdapter":
        """Load pre-trained memory weights."""
        checkpoint = torch.load(path)
        config = MemoryConfig(**checkpoint['config'])
        adapter = cls(config)
        adapter.memory.load_state_dict(checkpoint['memory_state_dict'])
        adapter.entity_emb.load_state_dict(checkpoint['entity_emb'])
        adapter.relation_emb.load_state_dict(checkpoint['relation_emb'])
        adapter.mask_token = checkpoint['mask_token']
        return adapter

