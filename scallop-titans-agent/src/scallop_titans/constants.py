"""
Shared constants for the Scallop-Titans Agent.
"""

# CLUTRR Relation Types
# Shared between TitansMemoryAdapter, ClutrrSequenceDataset, and TitansForeignPredicateFactory
CLUTRR_RELATIONS = [
    "aunt", "son-in-law", "grandfather", "brother", "sister",
    "father", "mother", "grandmother", "uncle", "daughter-in-law",
    "grandson", "granddaughter", "father-in-law", "mother-in-law",
    "nephew", "son", "daughter", "niece", "husband", "wife"
]

# Number of relation types (for embedding dimensions)
NUM_RELATIONS = len(CLUTRR_RELATIONS)

# Default entity vocabulary size (for hashing trick)
NUM_ENTITIES = 10000
