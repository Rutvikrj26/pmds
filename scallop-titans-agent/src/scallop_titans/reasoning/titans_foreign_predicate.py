"""
Titans Foreign Predicate Factory.

Creates a unified foreign predicate that queries Titans memory.
Allows Scallop to dynamically retrieve probabilistic facts for ANY relation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import torch
from torch import Tensor

# Graceful import for scallopy
try:
    from scallopy import foreign_predicate
    SCALLOPY_AVAILABLE = True
except ImportError:
    SCALLOPY_AVAILABLE = False
    # Mock decorator if not available
    def foreign_predicate(**kwargs):
        def decorator(func):
            return func
        return decorator

if TYPE_CHECKING:
    from scallop_titans.memory.titans_adapter import TitansMemoryAdapter
    import scallopy

class TitansForeignPredicateFactory:
    """
    Creates foreign predicates that query Titans memory.
    """
    
    def __init__(self, memory: TitansMemoryAdapter):
        self.memory = memory
    
    def register_all(self, ctx: scallopy.ScallopContext) -> None:
        """
        Register the unified foreign predicate.
        
        Registers:
            titans_query(relation: String, entity_a: String) -> String
            
        Yields:
            (probability_tag, entity_b)
        """
        
        @foreign_predicate(
            name="titans_query",
            input_arg_types=[str, str], # (relation, entity_a)
            output_arg_types=[str],     # (entity_b)
            # tag_type=Tensor # Leaving vague to rely on Scallop's inference for differentiable prov
        )
        def titans_query(relation: str, entity_a: str) -> Iterator[tuple[str]]:
            """
            Queries Titans memory: relation(entity_a, ?) -> ?
            Returns entities. Scallop assigns gradient-tracked tags automatically
            when using difftopkproofs provenance.
            """
            # Call the neural retrieval
            results = self.memory.retrieve_by_relation(relation, entity_a)
            
            for prob_tensor, entity_b in results:
                # Yield untagged output tuple
                # Note: With difftopkproofs, scallopy automatically assigns
                # gradient-tracked tensor tags (requires_grad=True).
                yield (entity_b,)
        
        ctx.register_foreign_predicate(titans_query)
