"""Tests for ClutrrTraceConverter."""
import json
from pathlib import Path
import pytest
from scallop_titans.data.clutrr_dataset import ClutrrSample, ClutrrSequenceDataset
from scallop_titans.data.trace_converter import ClutrrTraceConverter

class MockDataset:
    def get_samples(self):
        return [
            ClutrrSample(
                id="test1",
                story="Alice is Bob's mother. Bob is Charlie's father.",
                query=("Alice", "Charlie"),
                target="grandmother",
                fact_chain=[
                    ("Alice", "mother", "Bob"),
                    ("Bob", "father", "Charlie")
                ]
            )
        ]

def test_convert_sample():
    dataset = MockDataset()
    converter = ClutrrTraceConverter(dataset)
    
    trace = converter.convert_sample(dataset.get_samples()[0])
    
    # Verify Structure
    assert "messages" in trace
    msgs = trace["messages"]
    assert len(msgs) == 3
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"
    
    # Verify Content
    user_content = msgs[1]["content"]
    assert "Alice is Bob's mother" in user_content
    assert "Who is Alice's grandmother?" in user_content
    
    asst_content = msgs[2]["content"]
    assert "<|start_thought|>" in asst_content
    assert "<|call_scallop|>" in asst_content
    assert "add_fact(mother, Alice, Bob)" in asst_content
    assert "add_fact(father, Bob, Charlie)" in asst_content
    assert "query(grandmother, Alice, ?)" in asst_content
    assert "<|scallop_result|>" in asst_content
    assert "Charlie" in asst_content

def test_convert_all(tmp_path):
    dataset = MockDataset()
    converter = ClutrrTraceConverter(dataset)
    output_file = tmp_path / "test_output.jsonl"
    
    converter.convert_all(output_file)
    
    assert output_file.exists()
    with open(output_file) as f:
        line = f.readline()
        data = json.loads(line)
        assert len(data["messages"]) == 3
