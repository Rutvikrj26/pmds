"""
Negative Example Miner.

Generates hard negative samples (unanswerable queries) by:
1. Missing Information: Removing a critical link in the chain.
2. Contradictions: Adding facts that cause Scallop conflict (if engine supported it) or just broken logic.
   Actually, Scallop just returns empty result for contradictions usually unless defined.
   We focus on "Missing Info" -> Empty Result.

Output:
System: ... Call Scallop.
User: Story with missing link... Who is X's aunt?
Assistant: Thought... Call...
Result: []
Assistant: Based on the provided information, I cannot determine...
"""
import json
import random
from typing import Literal

from scallop_titans.data.trace_converter import ClutrrTraceConverter
from scallop_titans.data.synthetic_generator import SyntheticTraceGenerator, SyntheticConfig

class NegativeMiner:
    """Mines negative examples from synthetic traces."""
    
    def __init__(self):
        self.generator = SyntheticTraceGenerator()
        
    def mine_negatives(self, valid_traces_path: str, output_path: str, ratio: float = 0.25):
        """
        Read valid traces, perturb them to be broken, and save.
        Ratio: fraction of input traces to convert to negatives.
        """
        with open(valid_traces_path) as f:
            lines = f.readlines()
            
        num_negatives = int(len(lines) * ratio)
        if num_negatives == 0:
            return
            
        print(f"Mining {num_negatives} negatives from {len(lines)} valid traces...")
        
        count = 0
        with open(output_path, "w") as f_out:
            # Shuffle lines to pick random candidates
            candidates = lines[:]  # copy
            random.shuffle(candidates)
            
            for line in candidates:
                if count >= num_negatives:
                    break
                    
                trace = json.loads(line)
                neg_trace = self._perturb_trace(trace)
                
                if neg_trace:
                    f_out.write(json.dumps(neg_trace) + "\n")
                    count += 1
                    
        print(f"Saved {count} negatives to {output_path}")

    def _perturb_trace(self, trace: dict) -> dict | None:
        """
        Turn a positive trace into a negative one.
        Strategy: 'Broken Chain' - remove a middle fact.
        """
        # Parse trace to extract fact chain
        # Requires parsing the Assistant content or having the raw metadata.
        # The SFT trace format jsonified the messages.
        # It's hard to reverse engineer 'fact_chain' exactly from the string 
        # unless we stored metadata.
        #
        # Better approach: We should integrate generation of negatives INTO the generator pipeline
        # OR parsing "add_fact(parent, Alice, Bob)" from the string.
        # Let's parse the string. It's reliably formatted.
        
        try:
            msgs = trace["messages"]
            asst_content = msgs[2]["content"]
            
            # Extract Scallop command part
            start_tag = "<|call_scallop|>"
            end_tag = "<|end_thought|>" # Wait, checks format
            # Format: <|start_thought|>...<|call_scallop|>cmds<|end_thought|>...
            # The prompt in `trace_converter` is:
            # f"<|start_thought|>{thought}<|call_scallop|>{scallop_cmd}<|end_thought|>"
            
            p1 = asst_content.find(start_tag)
            p2 = asst_content.find(end_tag, p1)
            
            if p1 == -1 or p2 == -1:
                return None
                
            cmd_block = asst_content[p1 + len(start_tag) : p2]
            cmds = [c.strip() for c in cmd_block.split(".") if c.strip()]
            
            # Identify facts vs query
            facts = [c for c in cmds if c.startswith("add_fact")]
            query = [c for c in cmds if c.startswith("query")]
            
            if len(facts) < 2:
                # Can't break a 1-fact chain effectively without removing everything
                return None
                
            # STRATEGY: Remove a random fact (but not all)
            # This breaks the chain A->B->C into A->B, C. Link B->C missing.
            to_remove = random.randint(0, len(facts)-1)
            remaining_facts = facts[:to_remove] + facts[to_remove+1:]
            
            # Reconstruct Command
            new_cmd = ". ".join(remaining_facts + query)
            
            # Reconstruct Content
            # 1. Update Scallop Call
            new_content = asst_content[:p1 + len(start_tag)] + new_cmd + asst_content[p2:]
            
            # 2. Update Result to []
            # Find <|scallop_result|>...<|end_scallop_result|>
            r1 = new_content.find("<|scallop_result|>")
            r2 = new_content.find("<|end_scallop_result|>", r1)
            if r1 != -1 and r2 != -1:
                new_content = new_content[:r1 + len("<|scallop_result|>")] + "[]" + new_content[r2:]
            
            # 3. Update Final Answer
            # "Alice's aunt is Carol." -> "I cannot determine..."
            # Find end of result block
            r_end_tag = "<|end_scallop_result|>"
            r3 = new_content.find(r_end_tag)
            if r3 != -1:
                # Replace everything after tag
                prefix = new_content[:r3 + len(r_end_tag)]
                new_answer = " Based on the information provided, I cannot determine the relationship."
                new_content = prefix + new_answer
                
            # Update trace messages
            trace["messages"][2]["content"] = new_content
            
            # ALSO: Must update the User Story?
            # User msg: "Story... Question?"
            # If we removed a fact from the tool call, we MUST remove it from the story too!
            # Otherwise the model sees the info in text but fails to call the tool for it.
            # That teaches the model to ignore text! Bad.
            # We want: "Text is missing info -> Model calls partial facts -> Result empty -> Model says dunno."
            
            # Parsing story is hard.
            # "Alice is Bob's mother. Bob is Charlie's father."
            # We assume facts map to sentences.
            # We removed index `to_remove`.
            # Let's try to split story by sentences.
            user_content = msgs[1]["content"]
            # separate story from question
            # Assume question starts with "Who is..."
            q_start = user_content.rfind("Who is")
            if q_start == -1: return None
            
            story = user_content[:q_start].strip()
            question = user_content[q_start:]
            
            sentences = [s.strip() + "." for s in story.split(".") if s.strip()]
            
            if len(sentences) == len(facts):
                # Perfect alignment assumption
                remaining_sentences = sentences[:to_remove] + sentences[to_remove+1:]
                new_story = " ".join(remaining_sentences)
                trace["messages"][1]["content"] = f"{new_story} {question}"
                return trace
            else:
                # Mismatch (e.g. complex sentences). Skip to be safe.
                return None
                
        except Exception:
            return None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input valid traces")
    parser.add_argument("--output", required=True, help="Output negatives")
    parser.add_argument("--ratio", type=float, default=0.25)
    args = parser.parse_args()
    
    miner = NegativeMiner()
    miner.mine_negatives(args.input, args.output, args.ratio)

if __name__ == "__main__":
    main()
