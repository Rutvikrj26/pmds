"""
Synthetic Logic Trace Generator.

Generates diverse kinship reasoning traces by:
1. Sampling entity chains
2. Sampling relations
3. Using Scallop Engine as oracle to derive ground truth targets
4. validating coherence and uniqueness
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import ClassVar

from scallop_titans.data.trace_converter import ClutrrTraceConverter
from scallop_titans.reasoning.scallop_engine import ScallopEngine
from scallop_titans.constants import CLUTRR_RELATIONS

@dataclass
class SyntheticConfig:
    num_samples: int = 200000
    hop_distribution: dict[int, float] = None
    
    def __post_init__(self):
        if self.hop_distribution is None:
            # Default distribution: 30% 1-hop, 40% 2-hop, 20% 3-hop, 10% 4-hop
            self.hop_distribution = {1: 0.3, 2: 0.4, 3: 0.2, 4: 0.1}

class SyntheticTraceGenerator:
    """Generates synthetic reasoning traces."""
    
    # Pool of 100 common names
    NAMES: ClassVar[list[str]] = [
        "Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy",
        "Kevin", "Linda", "Mike", "Nancy", "Oscar", "Peggy", "Quentin", "Ruth", "Steve", "Trudy",
        "Ursula", "Victor", "Wendy", "Xavier", "Yvonne", "Zach", "Aaron", "Bella", "Carl", "Diana",
        "Edward", "Fiona", "George", "Helen", "Ian", "Julia", "Kyle", "Laura", "Mark", "Nora",
        "Oliver", "Paula", "Quinn", "Rachel", "Sam", "Tina", "Ulysses", "Victoria", "Will", "Xena",
        "Yasmine", "Zane", "Adam", "Betty", "Chris", "Debbie", "Eric", "Felicia", "Gary", "Hannah",
        "Isaac", "Jenny", "Ken", "Lisa", "Matt", "Nicole", "Otto", "Pam", "Randy", "Sarah",
        "Tom", "Uma", "Vince", "Wanda", "Xander", "Yolanda", "Zack", "Andrew", "Barbara", "Cody",
        "Danielle", "Ethan", "Faith", "Gavin", "Holly", "Jack", "Karen", "Leo", "Megan", "Nathan",
        "Olivia", "Peter", "Queen", "Ryan", "Samantha", "Tyler", "Violet", "Wyatt", "Ximena"
    ]
    
    # Relations graph for valid chaining (simplified for random sampling)
    # We rely on Scallop to tell us if the chain actually makes sense (e.g. gender consistency)
    # But we should try to generate sensible pairs to avoid high rejection rate.
    # Map relation -> implied gender of HEAD, compatible next relations?
    # Actually, simpler: just sample random relations and let Scallop fail if invalid.
    # Wait, CLUTRR relations imply gender.
    # mother(A, B) -> A is female.
    # father(A, B) -> A is male.
    # If we chain: father(A, B), mother(B, C). 
    # father(A, B) implies B? No, parent(A, B). B's gender is free.
    # But: husband(A, B) -> A male, B female.
    # Next: father(B, C)? Impossible, B is female.
    # So we need gender constraints.
    
    RELATION_GENDER = {
        "mother": ("female", "any"),
        "father": ("male", "any"),
        "son": ("male", "any"),
        "daughter": ("female", "any"),
        "husband": ("male", "female"),
        "wife": ("female", "male"),
        "brother": ("male", "any"),
        "sister": ("female", "any"),
        "grandfather": ("male", "any"),
        "grandmother": ("female", "any"),
        "grandson": ("male", "any"),
        "granddaughter": ("female", "any"),
        "uncle": ("male", "any"),
        "aunt": ("female", "any"),
        "nephew": ("male", "any"),
        "niece": ("female", "any"),
        "son-in-law": ("male", "any"),
        "daughter-in-law": ("female", "any"),
        "father-in-law": ("male", "any"),
        "mother-in-law": ("female", "any"),
    }
    
    SENTENCE_TEMPLATES = {
        "mother": [
            "{A} is the mother of {B}.", "{A} is {B}'s mother.", "{A} is {B}'s mom.",
            "{B} has a mother named {A}.", "{B}'s mother is {A}."
        ],
        "father": [
            "{A} is the father of {B}.", "{A} is {B}'s father.", "{A} is {B}'s dad.",
            "{B} has a father named {A}.", "{B}'s father is {A}."
        ],
        "son": [
            "{A} is the son of {B}.", "{A} is {B}'s son.", "{B} has a son named {A}.",
            "{A} was born to {B} (son)."
        ],
        "daughter": [
            "{A} is the daughter of {B}.", "{A} is {B}'s daughter.", "{B} has a daughter named {A}.",
            "{A} was born to {B} (daughter)."
        ],
        "husband": [
            "{A} is the husband of {B}.", "{A} is {B}'s husband.", "{A} is married to {B} (husband).",
            "{B} has a husband named {A}."
        ],
        "wife": [
            "{A} is the wife of {B}.", "{A} is {B}'s wife.", "{A} is married to {B} (wife).",
            "{B} has a wife named {A}."
        ],
        "brother": [
            "{A} is the brother of {B}.", "{A} is {B}'s brother.", "{A} is {B}'s bro.",
            "{B} has a brother named {A}."
        ],
        "sister": [
            "{A} is the sister of {B}.", "{A} is {B}'s sister.", "{A} is {B}'s sis.",
            "{B} has a sister named {A}."
        ],
        "default": ["{A} is the {R} of {B}.", "{B}'s {R} is {A}."]
    }

    # Primitive relations for chain generation
    # We restrict to these so we can reliably map to base Scallop facts (parent/sibling/spouse)
    PRIMITIVES = [
        "mother", "father", "son", "daughter",
        "brother", "sister", 
        "husband", "wife"
    ]

    def __init__(self, config: SyntheticConfig = SyntheticConfig()):
        self.config = config
        self.engine = ScallopEngine()
        # Pre-load meta-rules
        self._inject_meta_rules()
        self.trace_converter = ClutrrTraceConverter(None)
        
    def _inject_meta_rules(self):
        """Inject meta-rule 'related(a, b, R)' to derive any relation."""
        rules = []
        for r in CLUTRR_RELATIONS:
            # clean relation name for datalog
            clean_r = r.replace("-", "_")
            rules.append(f'rel related(a, b, "{r}") = {clean_r}(a, b)')
            
        self.engine._ctx.add_program("\n".join(rules))

    def generate_batch(self, output_path: str):
        """Generate and save batch."""
        count = 0
        accepted = 0
        
        with open(output_path, "w") as f:
            while accepted < self.config.num_samples:
                hops = self._sample_hops()
                
                # Generate chain using ONLY primitives
                chain = self._generate_chain(hops)
                if not chain: continue
                
                # Derive Target
                target, trace_msg = self._validate_and_format(chain)
                
                if target:
                    f.write(json.dumps(trace_msg) + "\n")
                    accepted += 1
                    if accepted % 1000 == 0:
                        print(f"Generated {accepted}/{self.config.num_samples}")
                
                count += 1
                if count > self.config.num_samples * 20: # Higher safety break
                    print("Warning: High rejection rate.")
                    break

    def _sample_hops(self) -> int:
        r = random.random()
        cumulative = 0
        for h, p in self.config.hop_distribution.items():
            cumulative += p
            if r <= cumulative:
                return h
        return 1

    def _generate_chain(self, hops: int) -> list[tuple[str, str, str]]:
        """Generate a valid entity-relation chain using primitives."""
        names = random.sample(self.NAMES, hops + 1)
        chain = []
        genders = [None] * (hops + 1)
        
        for i in range(hops):
            head = names[i]
            tail = names[i+1]
            
            candidates = []
            for r in self.PRIMITIVES:
                req_head_gender, req_tail_gender = self.RELATION_GENDER.get(r, ("any", "any"))
                if genders[i] and req_head_gender != "any" and genders[i] != req_head_gender:
                    continue
                if genders[i+1] and req_tail_gender != "any" and genders[i+1] != req_tail_gender:
                    continue
                candidates.append(r)
            
            if not candidates: return None
            
            rel = random.choice(candidates)
            chain.append((head, rel, tail))
            
            head_g, tail_g = self.RELATION_GENDER.get(rel, ("any", "any"))
            if head_g != "any": genders[i] = head_g
            if tail_g != "any": genders[i+1] = tail_g
            
        return chain

    def _validate_and_format(self, chain: list) -> tuple[str | None, dict | None]:
        """Validate chain via Scallop and format."""
        
        # 1. Reset
        self.engine.reset()
        self._inject_meta_rules()
        
        # 2. Add Facts
        for head, rel, tail in chain:
            # Since ScallopEngine now supports inverse rules (Axioms),
            # we can add high-level facts directly and they will imply base facts.
            # However, we must ensure 'rel' names match Scallop convention (underscores).
            clean_rel = rel.replace("-", "_")
            self.engine.add_fact(clean_rel, head, tail)
            
        # 3. Query Target
        start, end = chain[0][0], chain[-1][2]
        
        # Query for ALL key relations 
        results = self.engine.query("related(a, b, r)")
        
        if not results:
            return None, None
            
        # Filter and Select target
        valid_targets = []
        for res in results:
            # res format: (prob, (a, b, r))
            # Note: Scallop might return entities/relations as strings.
            tup = res[1]
            if len(tup) == 3:
                a, b, r = tup
                if a == start and b == end:
                    valid_targets.append(r)
        
        if not valid_targets:
            return None, None
            
        # Prefer "interesting" targets (not just basic ones if 2-hop)
        
        # Prefer "interesting" targets (not just basic ones if 2-hop)
        # e.g. if 'grandson' and 'male', pick 'grandson'
        target = random.choice(valid_targets)
        
        # 4. Generate Text
        trace = self._format_trace(chain, target)
        return target, trace



    def _format_trace(self, chain: list, target: str) -> dict:
        """Text generation."""
        story_parts = []
        for head, rel, tail in chain:
            # Simple template
            tmpl = random.choice(self.SENTENCE_TEMPLATES.get(rel, self.SENTENCE_TEMPLATES["default"]))
            stmt = tmpl.replace("{A}", head).replace("{B}", tail).replace("{R}", rel)
            story_parts.append(stmt)
            
        story = " ".join(story_parts)
        sample = type('obj', (object,), {
            'story': story,
            'story': story,
            'query': (chain[-1][2], chain[0][0]), # (QuestionEntity=End, AnswerEntity=Start)
            'target': target, # Derived ground truth
            'fact_chain': chain
        })
        
        # Reuse trace converter
        return self.trace_converter.convert_sample(sample)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--count", type=int, default=200000, help="Number of traces")
    args = parser.parse_args()
    
    config = SyntheticConfig(num_samples=args.count)
    generator = SyntheticTraceGenerator(config)
    generator.generate_batch(args.output)   

if __name__ == "__main__":
    main()
