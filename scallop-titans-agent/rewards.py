import re
from scallop_titans.reasoning.scallop_engine import ScallopEngine
from typing import List

# Global engine instance to avoid reloading rules every step
# (GRPO runs in process, so this works. Distributed might need care)
_ENGINE = None


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        try:
            _ENGINE = ScallopEngine()
        except Exception as e:
            print(f"Warning: Failed to init ScallopEngine: {e}")
            return None
    return _ENGINE


def format_reward_func(completions, **kwargs) -> List[float]:
    """
    Reward for using correct XML tags.
    +0.5 for <|start_thought|>
    +0.5 for <|call_scallop|>
    +0.5 for <|end_thought|>
    """
    rewards = []
    for completion in completions:
        score = 0.0
        if "<|start_thought|>" in completion:
            score += 0.5
        if "<|call_scallop|>" in completion:
            score += 0.5
        if "<|end_thought|>" in completion:
            score += 0.5
        rewards.append(score)
    return rewards


def syntax_reward_func(completions, **kwargs) -> List[float]:
    """
    Reward for valid Scallop syntax inside the block.
    Checks for add_fact(...) or query(...) patterns.
    """
    rewards = []
    # Engine unused here, relying on regex

    for completion in completions:
        score = 0.0
        # Extract content between <|call_scallop|> and <|end_thought|>
        match = re.search(
            r"<\|call_scallop\|>(.*?)<\|end_thought\|>", completion, re.DOTALL
        )
        if match:
            code = match.group(1)
            # Check for at least one valid command
            if re.search(r"add_fact\s*\(.+\)", code) or re.search(
                r"query\s*\(.+\)", code
            ):
                score += 1.0

            # Penalize hallucinations like "execute" or "run" if they aren't "run_query"
            if "execute " in code:
                score -= 0.5

        rewards.append(score)
    return rewards


def execution_reward_func(prompts, completions, **kwargs) -> List[float]:
    """
    Reward for successful execution of Scallop code.
    This runs the code using ScallopEngine.
    Result matching is harder without explicit ground truth passed here,
    so we focus on *runtime success* (no errors).
    """
    rewards = []
    engine = get_engine()

    for completion in completions:
        if not engine:
            rewards.append(0.0)
            continue

        match = re.search(
            r"<\|call_scallop\|>(.*?)<\|end_thought\|>", completion, re.DOTALL
        )
        if not match:
            rewards.append(0.0)
            continue

        cmd = match.group(1).strip()

        # Reset engine state for clean execution
        engine.reset()

        try:
            # We catch stdout/stderr or exceptions
            # ScallopEngine.execute_command returns a string result
            # Ideally we check if it returns "No results" or actual data
            result_str = engine.execute_command(cmd)

            if "Results:" in result_str:
                rewards.append(2.0)  # Bonus for finding results
            elif "Added:" in result_str:
                rewards.append(1.0)  # Correctly added facts
            else:
                rewards.append(0.5)  # Valid execution but no output

        except Exception:
            # Syntax error or runtime error
            rewards.append(-0.5)

    return rewards
