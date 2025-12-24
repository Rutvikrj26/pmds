"""Unit tests for TitansMemoryAdapter."""

import pytest
import torch

from scallop_titans.memory.titans_adapter import MemoryConfig, TitansMemoryAdapter


class TestTitansMemoryAdapter:
    """Tests for the Titans Memory Adapter."""

    @pytest.fixture
    def config(self) -> MemoryConfig:
        """Create a test configuration with small dimensions."""
        return MemoryConfig(
            dim=64,
            hidden_dim=32,
            num_layers=2,
            heads=2,
            dim_head=16,
            chunk_size=4,
        )

    @pytest.fixture
    def memory(self, config: MemoryConfig) -> TitansMemoryAdapter:
        """Create a memory adapter for testing."""
        return TitansMemoryAdapter(config)

    def test_init(self, memory: TitansMemoryAdapter) -> None:
        """Test memory initialization."""
        assert memory.config.dim == 64
        assert memory.memory is not None
        assert memory._state is None

    def test_store(self, memory: TitansMemoryAdapter) -> None:
        """Test storing facts in memory."""
        batch = 2
        dim = memory.config.dim

        entity_a = torch.randn(batch, dim)
        relation = torch.randn(batch, dim)
        entity_b = torch.randn(batch, dim)

        retrieved, surprise = memory.store(entity_a, relation, entity_b, return_surprise=True)

        # Check shapes
        assert retrieved.shape == (batch, 3, dim)
        assert memory._state is not None

    def test_retrieve(self, memory: TitansMemoryAdapter) -> None:
        """Test retrieving from memory."""
        batch = 2
        dim = memory.config.dim

        # Store something first
        entity_a = torch.randn(batch, dim)
        relation = torch.randn(batch, dim)
        entity_b = torch.randn(batch, dim)
        memory.store(entity_a, relation, entity_b)

        # Retrieve
        query = torch.randn(batch, dim)
        retrieved = memory.retrieve(query)

        assert retrieved.shape[0] == batch
        assert retrieved.shape[-1] == dim

    def test_retrieve_empty(self, memory: TitansMemoryAdapter) -> None:
        """Test retrieval when memory is empty."""
        batch = 2
        dim = memory.config.dim

        query = torch.randn(batch, dim)
        retrieved = memory.retrieve(query)

        # Should return zeros
        assert retrieved.shape == (batch, 1, dim)
        assert torch.allclose(retrieved, torch.zeros_like(retrieved))

    def test_reset_state(self, memory: TitansMemoryAdapter) -> None:
        """Test state reset."""
        # Store something
        batch, dim = 2, memory.config.dim
        memory.store(
            torch.randn(batch, dim),
            torch.randn(batch, dim),
            torch.randn(batch, dim),
        )
        assert memory._state is not None

        # Reset
        memory.reset_state()
        assert memory._state is None
        assert len(memory._surprise_history) == 0

    def test_apply_decay(self, memory: TitansMemoryAdapter) -> None:
        """Test weight decay application."""
        batch, dim = 2, memory.config.dim

        # Store to create state
        memory.store(
            torch.randn(batch, dim),
            torch.randn(batch, dim),
            torch.randn(batch, dim),
        )

        # No error on decay
        memory.apply_decay()


class TestMemoryChaining:
    """Test memory across multiple turns."""

    @pytest.fixture
    def memory(self) -> TitansMemoryAdapter:
        """Create memory for chaining tests."""
        config = MemoryConfig(dim=64, chunk_size=1, heads=2, dim_head=32)
        return TitansMemoryAdapter(config)

    # @pytest.mark.skip(reason="Dimension mismatch in NeuralMemory - needs investigation")
    def test_multi_turn_memory(self, memory: TitansMemoryAdapter) -> None:
        """Test memory persists across multiple turns."""
        batch, dim = 2, memory.config.dim

        # Turn 1
        memory.store(
            torch.randn(batch, dim),
            torch.randn(batch, dim),
            torch.randn(batch, dim),
        )
        state_after_1 = memory._state

        # Turn 2
        memory.store(
            torch.randn(batch, dim),
            torch.randn(batch, dim),
            torch.randn(batch, dim),
        )
        state_after_2 = memory._state

        # State should have progressed
        assert state_after_2 is not None
        assert state_after_2.seq_index > state_after_1.seq_index

    def test_memory_update_and_facts(self, memory: TitansMemoryAdapter) -> None:
        """Test parsing Scallop commands and retrieving facts."""
        cmd = "add_fact(parent, alice, bob). add_fact(sibling, bob, charlie)"
        
        # Update memory
        memory.update(cmd)
        
        # Check neural state was updated
        assert memory._state is not None
        assert len(memory.surprise_history) == 2  # 2 facts added
        
        # Check symbolic facts
        facts = memory.get_probabilistic_facts()
        assert len(facts) == 2
        assert ("alice", "parent", "bob", 0.95) in facts
        assert ("bob", "sibling", "charlie", 0.95) in facts
