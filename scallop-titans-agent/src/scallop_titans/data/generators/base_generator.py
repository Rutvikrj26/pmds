"""
Multi-Domain Data Generator: Base Classes and Registry.

This module provides the foundation for generating diverse logic reasoning
traces across 66 domains with 600+ unique relations.
"""
from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import ClassVar, Any


class RelationProperty(Enum):
    """Formal properties of relations for inference rules."""
    SYMMETRIC = auto()      # R(a,b) <=> R(b,a)
    TRANSITIVE = auto()     # R(a,b) ∧ R(b,c) => R(a,c)
    REFLEXIVE = auto()      # R(a,a) is always true
    IRREFLEXIVE = auto()    # R(a,a) is never true
    ANTISYMMETRIC = auto()  # R(a,b) ∧ R(b,a) => a=b


@dataclass
class Relation:
    """A typed relation with formal properties."""
    name: str
    head_type: str  # Entity type for first argument
    tail_type: str  # Entity type for second argument
    properties: set[RelationProperty] = field(default_factory=set)
    inverse: str | None = None  # Name of inverse relation if any
    
    # Sentence templates for natural language generation
    templates: list[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.templates:
            # Default template
            self.templates = [
                "{head} {name} {tail}.",
                "{tail} is {name} by {head}.",
            ]


@dataclass
class EntityType:
    """An entity type with a pool of possible names/identifiers."""
    name: str
    pool: list[str] = field(default_factory=list)
    prefix: str = ""  # For generated IDs like "Node-1", "Server-A"
    
    def sample(self, count: int, existing: set[str] | None = None) -> list[str]:
        """Sample unique entities from the pool or generate IDs."""
        existing = existing or set()
        
        if self.pool:
            available = [e for e in self.pool if e not in existing]
            if len(available) >= count:
                return random.sample(available, count)
            # If not enough in pool, supplement with generated
            result = available[:count]
            needed = count - len(result)
        else:
            result = []
            needed = count
            
        # Generate additional IDs
        idx = 1
        while len(result) < count:
            candidate = f"{self.prefix}{idx}" if self.prefix else f"{self.name}_{idx}"
            if candidate not in existing and candidate not in result:
                result.append(candidate)
            idx += 1
            
        return result


@dataclass
class DomainConfig:
    """Configuration for a specific domain."""
    name: str
    category: str
    relations: list[Relation]
    entity_types: dict[str, EntityType]
    
    # Scallop rules for this domain (inference rules)
    scallop_rules: list[str] = field(default_factory=list)
    
    # Hop distribution for chain generation
    hop_distribution: dict[int, float] = field(default_factory=lambda: {
        1: 0.25, 2: 0.35, 3: 0.25, 4: 0.10, 5: 0.05
    })
    
    # System prompt for this domain
    system_prompt: str = "You are a reasoning agent. Use <|call_scallop|> to invoke logic."


@dataclass
class GeneratedSample:
    """A generated training sample."""
    domain: str
    category: str
    story: str
    question: str
    facts: list[tuple[str, str, str]]  # (head, relation, tail)
    query_relation: str
    query_head: str
    query_tail: str  # Ground truth answer
    scallop_cmd: str
    scallop_result: str
    answer: str
    
    def to_chat_format(self, system_prompt: str = None) -> dict:
        """Convert to SFT chat format."""
        system = system_prompt or "You are a reasoning agent. Use <|call_scallop|> to invoke logic."
        
        # Generate thought
        thought = f"I need to find the {self.query_relation} of {self.query_head}."
        
        user_content = f"{self.story} {self.question}"
        
        assistant_content = (
            f"<|start_thought|>{thought}<|call_scallop|>{self.scallop_cmd}<|end_thought|>"
            f"<|scallop_result|>{self.scallop_result}<|end_scallop_result|>"
            f"{self.answer}"
        )
        
        return {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ],
            "metadata": {
                "domain": self.domain,
                "category": self.category,
                "hops": len(self.facts)
            }
        }


class BaseDomainGenerator(ABC):
    """Abstract base class for domain-specific generators."""
    
    # Subclasses should define these
    DOMAIN_NAME: ClassVar[str] = ""
    CATEGORY: ClassVar[str] = ""
    
    def __init__(self, config: DomainConfig | None = None):
        self.config = config or self._default_config()
        self._validate_config()
        
    @abstractmethod
    def _default_config(self) -> DomainConfig:
        """Return the default configuration for this domain."""
        pass
    
    def _validate_config(self):
        """Validate the domain configuration."""
        assert self.config.name, "Domain name required"
        assert self.config.relations, "At least one relation required"
        assert self.config.entity_types, "At least one entity type required"
        
    def generate_chain(self, hops: int) -> list[tuple[str, str, str]] | None:
        """
        Generate a valid relation chain with the specified number of hops.
        
        Returns None if generation fails (e.g., constraint violation).
        """
        # Sample entities
        all_types = list(self.config.entity_types.keys())
        used_entities: set[str] = set()
        chain: list[tuple[str, str, str]] = []
        
        # FIX: Only select from entity types that can START a chain
        # These are types that appear as head_type in at least one relation
        startable_head_types = {r.head_type for r in self.config.relations if r.head_type != "any"}
        valid_start_types = [t for t in all_types if t in startable_head_types]
        
        # Fallback: if "any" is used as head_type, all types are valid
        if not valid_start_types:
            has_any_head = any(r.head_type == "any" for r in self.config.relations)
            if has_any_head:
                valid_start_types = all_types
            else:
                return None  # No valid starting types exist
        
        # Start with a random entity from VALID start types
        first_type = random.choice(valid_start_types)
        entities = [self.config.entity_types[first_type].sample(1, used_entities)[0]]
        used_entities.add(entities[0])
        current_type = first_type
        
        for i in range(hops):
            # Find compatible relations based on current entity type
            compatible = [
                r for r in self.config.relations 
                if r.head_type == current_type or r.head_type == "any"
            ]
            
            if not compatible:
                return None  # Dead end
                
            rel = random.choice(compatible)
            
            # Get or create tail entity
            tail_type = rel.tail_type if rel.tail_type != "any" else random.choice(all_types)
            tail_entity = self.config.entity_types.get(
                tail_type, 
                self.config.entity_types[list(self.config.entity_types.keys())[0]]
            ).sample(1, used_entities)[0]
            used_entities.add(tail_entity)
            entities.append(tail_entity)
            
            chain.append((entities[-2], rel.name, tail_entity))
            current_type = tail_type
            
        return chain
    
    def chain_to_story(self, chain: list[tuple[str, str, str]]) -> str:
        """Convert a relation chain to natural language story."""
        sentences = []
        
        for head, rel_name, tail in chain:
            # Find the relation to get templates
            rel = next((r for r in self.config.relations if r.name == rel_name), None)
            if rel and rel.templates:
                template = random.choice(rel.templates)
                sentence = template.format(head=head, tail=tail, name=rel_name)
            else:
                sentence = f"{head} {rel_name.replace('_', ' ')} {tail}."
            sentences.append(sentence)
            
        return " ".join(sentences)
    
    def chain_to_scallop(self, chain: list[tuple[str, str, str]], 
                         query_rel: str, query_head: str) -> str:
        """Convert chain to Scallop command string."""
        cmds = []
        
        # Add facts
        for head, rel, tail in chain:
            cmds.append(f"add_fact({rel}, {head}, {tail})")
            
        # Add query
        cmds.append(f"query({query_rel}, {query_head}, ?)")
        
        return ". ".join(cmds)
    
    def generate_sample(self, negative_ratio: float = 0.1, distractor_ratio: float = 0.2) -> GeneratedSample | None:
        """
        Generate a single training sample.
        
        Args:
            negative_ratio: Probability of generating a negative (unanswerable) sample.
            distractor_ratio: Probability of injecting irrelevant facts (noise).
        """
        is_negative = random.random() < negative_ratio
        
        # Sample hop count
        hops = self._sample_hops()
        
        # Generate valid chain with retry logic
        # If multi-hop fails, we fall back to simpler chains
        chain = None
        max_attempts = 5
        for attempt in range(max_attempts):
            chain = self.generate_chain(hops)
            if chain:
                break
            # Reduce hop count on failure (fallback to simpler chains)
            hops = max(1, hops - 1)
        
        if not chain:
            return None

        # --- 1. Distractor Injection (Noise) ---
        # Add random facts that don't affect the main chain
        context_facts = list(chain)
        if random.random() < distractor_ratio:
            ignore_entities = set([c[0] for c in chain] + [c[2] for c in chain])
            # Generate a 1-hop disconnected chain
            distractor_chain = self.generate_chain(1)
            if distractor_chain:
                # Ensure it doesn't accidentally link to our main chain (simple heuristic)
                if distractor_chain[0][0] not in ignore_entities:
                     context_facts.extend(distractor_chain)
        
        # Shuffle facts so the model doesn't just read in order
        random.shuffle(context_facts)
            
        if is_negative:
            # Strategy: Broken Chain
            # We remove a specific link from the EVIDENCE and ask about IT.
            
            if len(chain) > 0:
                # Pick a random link to break (and query about)
                target_idx = random.randint(0, len(chain)-1)
                target_fact = chain[target_idx]
                
                # Query details come from the MISSING fact
                query_head = target_fact[0]
                query_rel = target_fact[1]
                query_tail = target_fact[2]
                
                # Remove this fact from context
                if target_fact in context_facts:
                    context_facts.remove(target_fact)
                    
                scallop_result = "[]"
                answer = "Based on the provided information, I cannot determine the relationship."
            else:
                return None # Should not happen with current logic

        else:
            # Positive Case: Pick a random link to query
            # The link MUST be present in context_facts (which might have distractors added)
            # We must pick from 'chain' since 'context_facts' has randomized order/distractors
            
            target_idx = random.randint(0, len(chain)-1)
            target_fact = chain[target_idx]
            
            query_head = target_fact[0]
            query_rel = target_fact[1]
            query_tail = target_fact[2]
            
            scallop_result = f"[('{query_tail}', 1.0)]"
            rel_phrase = self._format_relation_for_question(query_rel)
            answer = self._generate_answer(rel_phrase, query_head, query_tail)

        # --- Generate Text Components ---
        story = self.chain_to_story(context_facts)
        rel_phrase = self._format_relation_for_question(query_rel)
        question = self._generate_question(rel_phrase, query_head)
        
        # Scallop Command: Always add ALL context facts, then query
        scallop_cmd = self.chain_to_scallop(context_facts, query_rel, query_head)

        # --- 3. Detailed CoT Reasoning ---
        if is_negative:
            thought = (
                f"The user is asking for the '{rel_phrase}' of {query_head}. "
                f"I will check the available facts to see if "
                f"I can find this relationship. "
                f"I will query the '{query_rel}' relation."
            )
        else:
            thought = (
                f"The user wants to find the {rel_phrase} of {query_head}. "
                f"I see mention of {query_head} in the text. "
                f"I will use the '{query_rel}' relation to find the answer."
            )
            
        return GeneratedSample(
            domain=self.config.name,
            category=self.config.category,
            story=story,
            question=question,
            facts=context_facts,
            query_relation=query_rel,
            query_head=query_head,
            query_tail=query_tail,
            scallop_cmd=scallop_cmd,
            scallop_result=scallop_result,
            answer=answer
        )
    
    def _sample_hops(self) -> int:
        """Sample hop count from distribution, capped by structural limits."""
        max_hops = self._get_max_hops()
        
        # Sample from distribution
        r = random.random()
        cumulative = 0
        for h, p in self.config.hop_distribution.items():
            cumulative += p
            if r <= cumulative:
                # Cap at structural maximum
                return min(h, max_hops)
        return 1
    
    def _get_max_hops(self) -> int:
        """
        Determine maximum possible chain length based on domain structure.
        
        Multi-hop chains require some entity types to be both head AND tail types.
        If no such overlap exists, only 1-hop chains are possible.
        """
        head_types = {r.head_type for r in self.config.relations if r.head_type != "any"}
        tail_types = {r.tail_type for r in self.config.relations if r.tail_type != "any"}
        
        # Types that can continue a chain (appear as both head and tail)
        chainable_types = head_types & tail_types
        
        if not chainable_types:
            # No chainable types = only 1-hop possible
            return 1
        
        # With chainable types, multi-hop is possible
        # Return a reasonable max (5 is already the default distribution max)
        return 5
    
    def _format_relation_for_question(self, rel_name: str) -> str:
        """Format relation name for natural language."""
        # Handle common patterns
        phrase = rel_name.replace('_', ' ')
        
        # Remove redundant "of" patterns
        if phrase.endswith(' of'):
            phrase = phrase[:-3]
            
        return phrase
    
    def _generate_question(self, rel_phrase: str, head_entity: str) -> str:
        """Generate a natural question."""
        templates = [
            f"Who is the {rel_phrase} of {head_entity}?",
            f"What is {head_entity}'s {rel_phrase}?",
            f"Find the {rel_phrase} of {head_entity}.",
            f"Who does {head_entity} have as {rel_phrase}?",
        ]
        return random.choice(templates)
    
    def _generate_answer(self, rel_phrase: str, head: str, tail: str) -> str:
        """Generate a natural answer."""
        templates = [
            f"{head}'s {rel_phrase} is {tail}.",
            f"The {rel_phrase} of {head} is {tail}.",
            f"{tail} is {head}'s {rel_phrase}.",
        ]
        return random.choice(templates)
    
    def generate_batch(self, count: int, output_path: str | Path | None = None, 
                       negative_ratio: float = 0.1, distractor_ratio: float = 0.2) -> list[dict]:
        """Generate a batch of samples."""
        samples = []
        attempts = 0
        max_attempts = count * 10
        
        while len(samples) < count and attempts < max_attempts:
            sample = self.generate_sample(negative_ratio=negative_ratio, distractor_ratio=distractor_ratio)
            if sample:
                samples.append(sample.to_chat_format(self.config.system_prompt))
            attempts += 1
            
            if len(samples) % 100 == 0 and len(samples) > 0:
                print(f"[{self.config.name}] Generated {len(samples)}/{count}")
                
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")
            print(f"[{self.config.name}] Saved {len(samples)} samples to {path}")
            
        return samples
