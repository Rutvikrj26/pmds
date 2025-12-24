"""Tests for NegativeMiner."""
import json
import pytest
from scallop_titans.data.negative_miner import NegativeMiner

def test_perturb_trace():
    miner = NegativeMiner()
    
    # Mock trace
    trace = {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "Alice is Bob's mother. Bob is Charlie's father. Who is Alice's grandson?"},
            {"role": "assistant", "content": "<|start_thought|>...<|call_scallop|>add_fact(mother, Alice, Bob). add_fact(father, Bob, Charlie). query(grandson, Alice, ?)<|end_thought|><|scallop_result|>[('Charlie', 1.0)]<|end_scallop_result|>Alice's grandson is Charlie."}
        ]
    }
    
    neg_trace = miner._perturb_trace(trace)
    
    assert neg_trace is not None
    
    # Check User content (one sentence removed)
    user_content = neg_trace["messages"][1]["content"]
    assert "Who is Alice's grandson?" in user_content
    # Should have only 1 sentence remaining
    # Use loose check as we don't know which index was removed
    assert ("Alice is Bob's mother" in user_content) ^ ("Bob is Charlie's father" in user_content) # XOR
    
    # Check Assistant content
    asst_content = neg_trace["messages"][2]["content"]
    assert "[]" in asst_content # Empty result
    assert "I cannot determine" in asst_content
    
    # Check Scallop command (one fact removed)
    assert ("add_fact(mother" in asst_content) ^ ("add_fact(father" in asst_content)
