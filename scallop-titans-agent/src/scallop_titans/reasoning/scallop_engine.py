"""
ScallopEngine: Differentiable logic reasoning engine using Scallop.

Implements the Scallop Engine from master_plan.md Part A Section 2C:
- Datalog with negation, aggregation, and recursion
- Differentiable via provenance semiring (difftopkproofs)
- Rule bank for kinship (CLUTRR), transitivity, inheritance, math axioms
- Foreign predicate interface to connect with Titans memory
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scallop_titans.memory import TitansMemoryAdapter

# Try to import scallopy - will fail gracefully if not installed
try:
    import scallopy

    SCALLOPY_AVAILABLE = True
except ImportError:
    SCALLOPY_AVAILABLE = False
    scallopy = None  # type: ignore


@dataclass
class ScallopConfig:
    """Configuration for Scallop Engine."""

    # Provenance semiring for differentiability
    provenance: str = "difftopkproofs"  # Recommended from master_plan.md
    k: int = 3  # Top-k proofs to track

    # Rule files directory
    rules_dir: Path = field(default_factory=lambda: Path(__file__).parent / "rules")

    # Debug settings
    debug: bool = False


class ScallopEngine:
    """
    Differentiable logic reasoning engine using Scallop.

    This class provides:
    1. Loading and managing Datalog rules
    2. Adding probabilistic facts from Titans memory
    3. Querying with gradient flow back to fact sources
    4. Parsing Scallop command syntax from LLM output
    """

    def __init__(self, config: ScallopConfig | None = None) -> None:
        """
        Initialize the Scallop Engine.

        Args:
            config: Engine configuration. Uses defaults if None.

        Raises:
            ImportError: If scallopy is not installed.
        """
        if not SCALLOPY_AVAILABLE:
            raise ImportError(
                "scallopy is not installed. Install with: pip install scallopy\n"
                "Note: Scallopy requires Rust compiler for installation."
            )

        self.config = config or ScallopConfig()

        # Create Scallop context with differentiable provenance
        # k is passed as a separate parameter to ScallopContext
        self._ctx = scallopy.ScallopContext(
            provenance=self.config.provenance,
            k=self.config.k
        )
        
        # Foreign Predicate Factory
        self._titans_fp_factory = None

        # Load default rules (can be customized)
        self._load_rules()

        # Track facts for debugging/reset
        self._facts: list[tuple] = []
        
    def _load_rules(self, rules_files: list[str] = ["common.scl", "kinship.scl"]) -> None:
        """Load Datalog rules from files."""
        # Define entity type for safety domains
        self._ctx.add_relation("entity", (str,))

        for filename in rules_files:
            path = self.config.rules_dir / filename
            if path.exists():
                with open(path) as f:
                    self._ctx.add_program(f.read())
            else:
                # Fallback or warning?
                # For Phase 3, we expect them to exist.
                print(f"Warning: Rule file {path} not found.")

    def set_titans_memory(self, memory: "TitansMemoryAdapter") -> None:
        """
        Bind Titans memory for foreign predicate queries.
        
        This enables the engine to dynamically query the neural memory
        during reasoning (e.g. for titans_parent(X)).
        """
        from scallop_titans.reasoning.titans_foreign_predicate import TitansForeignPredicateFactory
        from scallop_titans.constants import CLUTRR_RELATIONS
        
        self._titans_fp_factory = TitansForeignPredicateFactory(memory)
        self._titans_fp_factory.register_all(self._ctx)
        
        # Dynamic Bridge Rules
        # Connect specific Scallop relations to the generic titans_query predicate
        # Rule: rel {relation}(a, b) = titans_query("{relation}", a, b)
        # Note: We sanitize relation names for Datalog (hyphens -> underscores)
        bridge_rules = []
        for rel in CLUTRR_RELATIONS:
            clean_rel = rel.replace("-", "_")
            # Scallop string literals need quotes
            rule = f'rel {clean_rel}(a, b) = titans_query("{rel}", a, b)'
            bridge_rules.append(rule)
            
        if bridge_rules:
            # We must ensure 'a' is bound. 
            # We assume a type 'entity(String)' exists and is populated with relevant entities.
            # Rule: rel father(a, b) = entity(a) and titans_query("father", a, b)
            safe_bridge_rules = []
            for rule in bridge_rules:
                # Insert entity(a) constraint
                # rule format: rel name(a, b) = titans_query(...)
                parts = rule.split(" = ")
                head = parts[0]
                body = parts[1]
                safe_rule = f"{head} = entity(a) and {body}"
                safe_bridge_rules.append(safe_rule)
                
            self._ctx.add_program("\n".join(safe_bridge_rules))



    def add_fact(
        self,
        relation: str,
        *args: str,
        probability: float = 1.0,
    ) -> None:
        """
        Add a probabilistic fact to the knowledge base.

        Args:
            relation: The relation name (e.g., "parent", "sibling")
            *args: The entities involved in the relation
            probability: Probability of the fact (0.0 to 1.0)
        """
        import torch
        
        # Track for debugging
        self._facts.append((relation, args, probability))

        # Add to Scallop context
        # For differentiable provenance, we need tensor tags
        if "diff" in self.config.provenance:
            tag = torch.tensor(probability, requires_grad=True)
            self._ctx.add_facts(relation, [(tag, args)])
        elif probability < 1.0:
            # Probabilistic fact with float tag
            self._ctx.add_facts(relation, [(probability, args)])
        else:
            # Certain fact (no tag needed)
            self._ctx.add_facts(relation, [args])

    def add_facts_from_memory(
        self,
        memory: "TitansMemoryAdapter",
    ) -> None:
        """
        Import probabilistic facts from Titans memory.

        This is the bridge between neural memory and symbolic reasoning.

        Args:
            memory: The Titans memory adapter to import facts from.
        """
        facts = memory.get_probabilistic_facts()
        for entity_a, relation, entity_b, prob in facts:
            self.add_fact(relation, entity_a, entity_b, probability=prob)

    def query(
        self,
        query_str: str,
        fact_source: "TitansMemoryAdapter | None" = None,
    ) -> list[tuple[Any, ...]]:
        """
        Execute a query against the knowledge base.

        Args:
            query_str: The query string (e.g., "aunt(?, alice)")
            fact_source: Optional Titans memory to import facts from first.

        Returns:
            List of result tuples with probabilities.
        """
        # Import facts from memory if provided
        if fact_source is not None:
            self.add_facts_from_memory(fact_source)

        # Parse the query to extract relation and arguments
        parsed = self._parse_query(query_str)
        if parsed is None:
            return []

        relation, args = parsed

        # Run the query
        self._ctx.run()

        # Get results
        try:
            results = list(self._ctx.relation(relation))
            return results
        except Exception as e:
            if self.config.debug:
                print(f"Query error: {e}")
            return []

    def _parse_query(self, query_str: str) -> tuple[str, list[str | None]] | None:
        """
        Parse a query string into relation and arguments.

        Args:
            query_str: Query like "aunt(alice, ?)" or "query(aunt, alice, ?)"

        Returns:
            Tuple of (relation_name, [args]) or None if parse fails.
        """
        # Handle "query(relation, arg1, arg2, ...)" format
        query_match = re.match(r"query\s*\(\s*(\w+)\s*,\s*(.+)\s*\)", query_str.strip())
        if query_match:
            relation = query_match.group(1)
            args_str = query_match.group(2)
            args = [a.strip() if a.strip() != "?" else None for a in args_str.split(",")]
            return relation, args

        # Handle "relation(arg1, arg2, ...)" format
        rel_match = re.match(r"(\w+)\s*\(\s*(.+)\s*\)", query_str.strip())
        if rel_match:
            relation = rel_match.group(1)
            args_str = rel_match.group(2)
            args = [a.strip() if a.strip() != "?" else None for a in args_str.split(",")]
            return relation, args

        return None

    def parse_scallop_command(self, cmd: str) -> list[dict[str, Any]]:
        """
        Parse a Scallop command from LLM output.

        Handles commands like:
        - add_fact(parent, alice, betty)
        - query(aunt, alice, ?)

        Args:
            cmd: The command string from LLM.

        Returns:
            List of parsed command dicts.
        """
        commands = []

        # Split by '.' to handle multiple commands
        for part in cmd.split("."):
            part = part.strip()
            if not part:
                continue

            # Parse add_fact
            add_match = re.match(r"add_fact\s*\(\s*(\w+)\s*,\s*(.+)\s*\)", part)
            if add_match:
                relation = add_match.group(1)
                args = [a.strip() for a in add_match.group(2).split(",")]
                commands.append({
                    "type": "add_fact",
                    "relation": relation,
                    "args": args,
                })
                continue

            # Parse query
            # Supports: query(relation, arg1, arg2...)
            # We translate to Datalog: relation("arg1", "arg2")
            # Handling '?' as variable 'var'
            query_match = re.match(r"query\s*\(\s*(.+)\s*\)", part)
            if query_match:
                content = query_match.group(1)
                # Split by comma
                tokens = [t.strip() for t in content.split(",")]
                if len(tokens) >= 1:
                    relation = tokens[0]
                    args = []
                    for arg in tokens[1:]:
                        if arg == "?":
                            args.append("var")
                        else:
                            # Quote string constants
                            args.append(f'"{arg}"')
                    
                    datalog_query = f"{relation}({', '.join(args)})"
                    
                    commands.append({
                        "type": "query",
                        "query": datalog_query,
                    })

        return commands

    def execute_command(
        self,
        cmd: str,
        memory: "TitansMemoryAdapter | None" = None,
    ) -> str:
        """
        Execute a Scallop command and return formatted results.

        This is the main entry point called by the agent when
        <|call_scallop|> is detected.

        Args:
            cmd: The Scallop command string.
            memory: Optional Titans memory for fact retrieval.

        Returns:
            Formatted result string for injection back into LLM.
        """
        commands = self.parse_scallop_command(cmd)
        results = []

        for command in commands:
            if command["type"] == "add_fact":
                self.add_fact(command["relation"], *command["args"])
                results.append(f"Added: {command['relation']}({', '.join(command['args'])})")

            elif command["type"] == "query":
                query_results = self.query(command["query"], fact_source=memory)
                if query_results:
                    formatted = [f"({', '.join(map(str, r))})" for r in query_results]
                    results.append(f"Results: {formatted}")
                else:
                    results.append("No results found")

        return "; ".join(results) if results else "No commands executed"

    def reset(self) -> None:
        """Reset the engine state (clear all facts)."""
        self._ctx = scallopy.ScallopContext(
            provenance=self.config.provenance,
            k=self.config.k
        )
        self._load_rules()
        self._facts.clear()

    @property
    def facts(self) -> list[tuple[str, tuple, float]]:
        """Get all added facts for debugging."""
        return self._facts.copy()
