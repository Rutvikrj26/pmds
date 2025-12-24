"""
Trace Converter for CLUTRR.

Converts CLUTRR samples (story, facts, query, target) into SFT-ready chat logs
with Scallop tool calls and CoT reasoning.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from scallop_titans.data.clutrr_dataset import ClutrrSample, ClutrrSequenceDataset


class ClutrrTraceConverter:
    """Convert CLUTRR samples to Scallop reasoning traces."""
    
    THOUGHT_TEMPLATES = [
        "I need to determine the {TARGET} of {SOURCE}.",
        "To find the {TARGET}, I will map the family tree starting from {SOURCE}.",
        "I will extract the relationships mentioned in the story to find who is the {TARGET} of {SOURCE}.",
        "I need to figure out how {SOURCE} is related to {TARGET_ENTITY} to identify the {TARGET}.",
        "Using the family connections described, I will trace the path from {SOURCE}.",
    ]
    
    def __init__(self, dataset: ClutrrSequenceDataset):
        self.dataset = dataset
        
    def convert_sample(self, sample: ClutrrSample) -> dict:
        """
        Convert a single CLUTRR sample to SFT chat format.
        
        Input: ClutrrSample(story, query, target, fact_chain)
        Output: {"messages": [...]} for SFT
        
        Format:
        System: You are a reasoning agent. Use <|call_scallop|> to invoke logic.
        User: {story} Who is {query_head}'s {target_rel}? (or derived natural question)
        Assistant: <|start_thought|>{thought}<|call_scallop|>{scallop_cmd}<|end_thought|>
                   <|scallop_result|>{result}<|end_scallop_result|>
                   {answer}
        """
        # 1. User Message
        user_msg = self._format_user_message(sample)
        
        # 2. Scallop Command
        scallop_cmd = self._facts_to_scallop(sample.fact_chain, sample.query, sample.target)
        
        # 3. Thought
        thought = self._generate_thought(sample.query[0], sample.target, sample.query[1])
        
        # 4. Result (Simulated for CLUTRR since we know the ground truth target)
        # Note: In real inference, Scallop produces this. For SFT data, we mock the expected output.
        # Format: [(target, 1.0)] or similar. Since CLUTRR target is the relationship name,
        # but the query is "query(aunt, alice, ?)", the result should be the ENTITY.
        # However, CLUTRR dataset 'target' field is the RELATION (e.g. 'aunt').
        # The query logic in existing code is generic: query(relation, head, ?) -> tail
        # But wait, CLUTRR task is usually: "How is A related to B?" -> "aunt"
        # OR "Who is A's aunt?" -> "B"
        # Let's check ClutrrSample definition.
        # sample.target is text like "aunt".
        # sample.query is (Head, Tail).
        # So the question is "What is relation between Head and Tail?"
        # Scallop query should probably be `related(Head, Tail, R)` if we use the meta-rule logic,
        # OR we can frame it as "Who is the {target} of {Head}?" but that gives away the target in the query!
        
        # Actually in `implementation_plan` we decided:
        # "Who is Alice's aunt?" -> query(aunt, alice, ?) -> result: [(carol, 0.98)]
        # This implies we KNOW the target relation is "aunt" from the question.
        # But in CLUTRR, the question is often "How is X related to Y?".
        # If the user asks "How is X related to Y?", the LLM can't guess "query(aunt...)" without knowing the answer.
        # So we should probably train the LLM to use the meta-rule: `related(X, Y, R)`?
        # OR, we stick to the plan's user query: "Who is Alice's aunt?".
        
        # Let's see `_format_user_message`.
        # If we construct the question as "Who is A's {target}?", then the LLM knows to query {target}.
        # This is valid for training tool use mechanics.
        
        scallop_result = f"[('{sample.query[1]}', 1.0)]"
        
        # 5. Answer
        answer = f"{sample.query[0]}'s {sample.target} is {sample.query[1]}."
        
        # Construct Assistant Content
        assistant_content = (
            f"<|start_thought|>{thought}<|call_scallop|>{scallop_cmd}<|end_thought|>"
            f"<|scallop_result|>{scallop_result}<|end_scallop_result|>"
            f"{answer}"
        )
        
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a reasoning agent. Use <|call_scallop|> to invoke logic."
                },
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_content}
            ]
        }
        
    def _format_user_message(self, sample: ClutrrSample) -> str:
        """Combine story and question."""
        # Story is already clean text
        # Formulate question: "Who is {Head}'s {Target}?"
        # This simplifies the task to "Extract facts and query specific relation"
        # rather than "Discover relation".
        # For our agentic goal (tool use), this is a good first step.
        question = f"Who is {sample.query[0]}'s {sample.target}?"
        return f"{sample.story} {question}"
        
    def _generate_thought(self, source: str, target_rel: str, target_entity: str) -> str:
        """Generate Chain-of-Thought text."""
        template = random.choice(self.THOUGHT_TEMPLATES)
        return template.format(SOURCE=source, TARGET=target_rel, TARGET_ENTITY=target_entity)
        
    def _facts_to_scallop(self, fact_chain: list[tuple[str, str, str]], query: tuple[str, str], target: str) -> str:
        """Convert facts to Scallop command string."""
        cmds = []
        
        # 1. Add Facts
        for head, rel, tail in fact_chain:
            # simple sanitization if needed
            cmds.append(f"add_fact({rel}, {head}, {tail})")
            
        # 2. Query
        # matching "Who is Head's Target?"
        # We need to find X such that target(X, Head) is true.
        # e.g. target="aunt", head="Ethan". We want X where aunt(X, Ethan).
        # Syntax: query(target, ?, head)
        cmds.append(f"query({target}, ?, {query[0]})")
        
        return ". ".join(cmds)
        
    def convert_all(self, output_path: str | Path):
        """Convert all samples and save to JSONL."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        samples = self.dataset.get_samples()
        print(f"Converting {len(samples)} samples...")
        
        with open(path, "w") as f:
            for s in samples:
                trace = self.convert_sample(s)
                f.write(json.dumps(trace) + "\n")
                
        print(f"Saved to {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CLUTRR csv")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()
    
    dataset = ClutrrSequenceDataset(args.csv)
    converter = ClutrrTraceConverter(dataset)
    converter.convert_all(args.output)


if __name__ == "__main__":
    main()
