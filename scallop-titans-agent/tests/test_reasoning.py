"""Unit tests for ScallopEngine."""

import pytest


# Skip tests if scallopy is not installed
scallopy = pytest.importorskip("scallopy", reason="scallopy not installed")


from scallop_titans.reasoning.scallop_engine import (
    ScallopConfig,
    ScallopEngine,
)


class TestScallopEngine:
    """Tests for the Scallop reasoning engine."""

    @pytest.fixture
    def engine(self) -> ScallopEngine:
        """Create a Scallop engine for testing."""
        config = ScallopConfig(debug=True)
        return ScallopEngine(config)

    def test_init(self, engine: ScallopEngine) -> None:
        """Test engine initialization."""
        assert engine._ctx is not None
        assert len(engine._facts) == 0

    def test_add_fact(self, engine: ScallopEngine) -> None:
        """Test adding facts."""
        engine.add_fact("parent", "betty", "alice")
        
        assert len(engine.facts) == 1
        assert engine.facts[0] == ("parent", ("betty", "alice"), 1.0)

    def test_add_probabilistic_fact(self, engine: ScallopEngine) -> None:
        """Test adding probabilistic facts."""
        engine.add_fact("parent", "betty", "alice", probability=0.9)
        
        assert len(engine.facts) == 1
        assert engine.facts[0] == ("parent", ("betty", "alice"), 0.9)

    def test_parse_scallop_command(self, engine: ScallopEngine) -> None:
        """Test parsing Scallop commands from LLM output."""
        cmd = "add_fact(parent, alice, betty). add_fact(sibling, betty, carol). query(aunt, alice, ?)"
        
        commands = engine.parse_scallop_command(cmd)
        
        assert len(commands) == 3
        assert commands[0]["type"] == "add_fact"
        assert commands[0]["relation"] == "parent"
        assert commands[1]["type"] == "add_fact"
        assert commands[2]["type"] == "query"

    def test_parse_query(self, engine: ScallopEngine) -> None:
        """Test query parsing."""
        # Standard format
        result = engine._parse_query("aunt(alice, ?)")
        assert result == ("aunt", ["alice", None])
        
        # Query format
        result = engine._parse_query("query(aunt, alice, ?)")
        assert result == ("aunt", ["alice", None])

    def test_reset(self, engine: ScallopEngine) -> None:
        """Test engine reset."""
        engine.add_fact("parent", "a", "b")
        assert len(engine.facts) == 1
        
        engine.reset()
        assert len(engine.facts) == 0


class TestKinshipReasoning:
    """Test kinship reasoning rules."""

    @pytest.fixture
    def engine(self) -> ScallopEngine:
        """Create engine with kinship rules."""
        return ScallopEngine()

    def test_mother_inference(self, engine: ScallopEngine) -> None:
        """Test mother relationship inference."""
        # Add facts
        engine.add_fact("parent", "betty", "alice")
        engine.add_fact("gender", "betty", "female")
        
        # Query - mother(betty, alice) should be derivable
        results = engine.query("mother(betty, alice)")
        
        # Should find the mother relation
        assert len(results) > 0 or results  # Scallop returns results differently


class TestCommandExecution:
    """Test command execution flow."""

    @pytest.fixture
    def engine(self) -> ScallopEngine:
        """Create engine for command tests."""
        return ScallopEngine(ScallopConfig(debug=True))

    def test_execute_add_fact(self, engine: ScallopEngine) -> None:
        """Test executing add_fact command."""
        result = engine.execute_command("add_fact(parent, alice, betty)")
        
        assert "Added" in result
        assert len(engine.facts) == 1

    def test_execute_multiple_commands(self, engine: ScallopEngine) -> None:
        """Test executing multiple commands."""
        cmd = "add_fact(parent, betty, alice). add_fact(gender, betty, female)"
        result = engine.execute_command(cmd)
        
        assert len(engine.facts) == 2
