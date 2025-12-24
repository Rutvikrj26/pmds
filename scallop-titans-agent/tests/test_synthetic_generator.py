"""Tests for SyntheticTraceGenerator."""
import json
import pytest
from scallop_titans.data.synthetic_generator import SyntheticTraceGenerator, SyntheticConfig

def test_generate_chain():
    config = SyntheticConfig(num_samples=10)
    gen = SyntheticTraceGenerator(config)
    
    # Test 2-hop chain
    chain = gen._generate_chain(2)
    assert len(chain) == 2
    # Verify entity linking
    # A->B, B->C
    assert chain[0][2] == chain[1][0]
    
def test_validation_logic():
    """Test that Scallop correctly validates a known valid chain."""
    gen = SyntheticTraceGenerator()
    
    # father(A, B), father(B, C) -> grandfather(A, C)
    chain = [("Alice", "father", "Bob"), ("Bob", "father", "Charlie")]
    
    target, trace = gen._validate_and_format(chain)
    
    assert target == "grandfather"
    assert trace is not None
    assert "messages" in trace
    assert "Alice" in trace["messages"][1]["content"]
    assert "Charlie" in trace["messages"][1]["content"]

def test_generate_batch(tmp_path):
    output = tmp_path / "synthetic.jsonl"
    config = SyntheticConfig(num_samples=5)
    gen = SyntheticTraceGenerator(config)
    
    gen.generate_batch(str(output))
    
    assert output.exists()
    lines = output.read_text().strip().split("\n")
    assert len(lines) == 5
    
    # Check JSON validity
    data = json.loads(lines[0])
    assert "messages" in data
