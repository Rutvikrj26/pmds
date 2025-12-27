#!/usr/bin/env python3
"""
Test Plumbing

Verifies the integration between Titans Memory and Scallop Engine explicitly.
Does NOT load the LLM (fast).
"""

import sys
from scallop_titans.memory import TitansMemoryAdapter
from scallop_titans.reasoning import ScallopEngine

def test_plumbing():
    print("Initializing components...")
    try:
        titans = TitansMemoryAdapter()
        scallop = ScallopEngine()
        scallop.set_titans_memory(titans)
    except Exception as e:
        print(f"FAILED: Component init: {e}")
        return False

    print("Testing Titans Memory Store...")
    try:
        # Simulate an LLM command update
        cmd = "add_fact(parent, alice, bob)"
        titans.update(cmd)
        facts = titans.get_probabilistic_facts()
        print(f"Titans Facts: {facts}")
        assert len(facts) == 1
        assert facts[0][0] == "alice"
        assert facts[0][1] == "parent"
        assert facts[0][2] == "bob"
    except Exception as e:
        print(f"FAILED: Titans update: {e}")
        return False

    print("Testing Scallop Query with Foreign Predicate...")
    try:
        # Query Scallop - should pull from Titans
        # Expected: parent(alice, bob) -> valid
        results = scallop.query("query(parent, alice, ?)", fact_source=titans)
        print(f"Scallop Results: {results}")
        
        # Check if we got ('bob',)
        # Note: Scallop returns tuples
        found = False
        for r in results:
            if "bob" in str(r):
                found = True
                break
        
        if not found:
             print("FAILED: Did not find 'bob' in results.")
             return False
             
    except Exception as e:
        print(f"FAILED: Scallop query: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    print("\n✅ Plumbing Test SUCCESS")
    return True

if __name__ == "__main__":
    success = test_plumbing()
    sys.exit(0 if success else 1)
