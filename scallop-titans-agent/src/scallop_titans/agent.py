"""
ScallopTitansAgent: Main agent class combining LLM, Titans Memory, and Scallop Engine.

This implements the inference loop from master_plan.md Part D Section 2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from scallop_titans.memory import TitansMemoryAdapter
    from scallop_titans.reasoning import ScallopEngine


@dataclass
class ChatMessage:
    """Single message in conversation history."""

    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class AgentConfig:
    """Configuration for ScallopTitansAgent."""

    # LLM settings
    model_name: str = "Qwen/Qwen3-32B"
    max_new_tokens: int = 2048
    temperature: float = 0.7

    # Special tokens (leveraging Qwen3's native think tokens)
    call_scallop_token: str = "<|call_scallop|>"
    scallop_result_token: str = "<|scallop_result|>"
    end_scallop_result_token: str = "<|end_scallop_result|>"
    start_thought_token: str = "<|start_thought|>"
    end_thought_token: str = "<|end_thought|>"

    # Memory settings
    memory_hidden_dim: int = 256
    memory_layers: int = 2
    memory_decay_rate: float = 0.01

    # Session settings
    serialize_memory: bool = False  # If True, save weights to disk per turn


@dataclass
class AgentState:
    """Holds session state for the agent."""

    history: list[ChatMessage] = field(default_factory=list)
    titans_weights: dict | None = None
    turn_count: int = 0


class ScallopTitansAgent:
    """
    Main agent combining LLM, Titans Memory, and Scallop Engine.

    Architecture from master_plan.md:
    - LLM generates text including <call_scallop> tokens
    - Titans Memory stores facts with surprise-based updates
    - Scallop Engine performs differentiable logic reasoning
    - Results are injected back into LLM for continued generation
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        llm: torch.nn.Module | None = None,
        tokenizer: Any | None = None,
        titans_memory: "TitansMemoryAdapter | None" = None,
        scallop_engine: "ScallopEngine | None" = None,
    ) -> None:
        """
        Initialize the ScallopTitansAgent.

        Args:
            config: Agent configuration. Uses defaults if None.
            llm: Pre-loaded LLM. Will be loaded based on config if None.
            tokenizer: Pre-loaded tokenizer.
            titans_memory: Pre-initialized Titans memory. Created from config if None.
            scallop_engine: Pre-initialized Scallop engine. Created if None.
        """
        self.config = config or AgentConfig()
        self.llm = llm
        self.tokenizer = tokenizer
        self.titans = titans_memory
        self.scallop = scallop_engine
        self._state = AgentState()

        # Load model/tokenizer if not provided but configured (placeholder for now)
        if self.llm is None and self.config.model_name and self.tokenizer is None:
            # We don't auto-load here to avoid heavy operations in init without explicit intent
            pass

    def _build_prompt(self, user_message: str) -> str:
        """
        Build the full prompt from history and new message.

        Args:
            user_message: The new user message to respond to.

        Returns:
            Formatted prompt string for the LLM.
        """
        # System message for reasoning agent
        system_prompt = (
            "You are a reasoning agent. Use <|call_scallop|> to invoke logic "
            "when you need to reason about relationships, kinship, or perform "
            "multi-hop inference. Think step by step."
        )

        # Build conversation
        messages = [f"System: {system_prompt}"]

        for msg in self._state.history:
            messages.append(f"{msg.role.capitalize()}: {msg.content}")

        messages.append(f"User: {user_message}")
        messages.append("Assistant: ")

        return "\n\n".join(messages)

    def _extract_scallop_cmd(self, response: str) -> str | None:
        """
        Extract Scallop command from response text.

        Args:
            response: The LLM response text.

        Returns:
            The Scallop command string, or None if not found.
        """
        pattern = rf"{re.escape(self.config.call_scallop_token)}(.+?)(?:{re.escape(self.config.end_thought_token)}|$)"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def chat(self, user_message: str) -> str:
        """
        Process a user message and generate a response.

        Implements the inference loop from master_plan.md Part D Section 2:
        1. Build prompt from history
        2. Generate LLM response (may contain tool calls)
        3. If <|call_scallop|> detected:
           a. Extract Scallop command
           b. Update Titans memory
           c. Query Scallop with Titans as fact source
           d. Inject result and continue generation
        4. Return final answer

        Args:
            user_message: The user's input message.

        Returns:
            The agent's response after reasoning.
        """
        # Add user message to history
        self._state.history.append(ChatMessage(role="user", content=user_message))
        self._state.turn_count += 1

        # Build initial prompt
        prompt = self._build_prompt(user_message)

        # Generation loop
        response = ""
        max_iterations = 10  # Prevent infinite loops

        for _ in range(max_iterations):
            # Generate next chunk 
            chunk = self._generate_chunk(prompt + response)
            response += chunk

            # Check for tool call
            if self.config.call_scallop_token in chunk:
                # Extract command
                scallop_cmd = self._extract_scallop_cmd(response)

                if scallop_cmd and self.titans:
                    # Update Titans memory (may add new facts)
                    # Note: ScallopEngine needs to be available to effectively use the updated memory
                    self.titans.update(scallop_cmd, self._state.titans_weights)

                    if self.scallop:
                        # Query Scallop with Titans as fact source
                        result = self.scallop.query(scallop_cmd, fact_source=self.titans)

                        # Inject result
                        response += (
                            f"{self.config.scallop_result_token}{result}"
                            f"{self.config.end_scallop_result_token}"
                        )
                    else:
                        response += f"{self.config.scallop_result_token}Error: Scallop Engine not available{self.config.end_scallop_result_token}"
            else:
                # No more tool calls, exit loop
                break

        # Extract final answer (after last end_thought token if present)
        final_answer = response
        if self.config.end_thought_token in response:
            final_answer = response.split(self.config.end_thought_token)[-1].strip()

        # Add assistant response to history
        self._state.history.append(ChatMessage(role="assistant", content=response))

        return final_answer

    def _generate_chunk(self, prompt: str) -> str:
        """
        Generate text chunk from LLM.

        Uses transformers generation if LLM and tokenizer are available.

        Args:
            prompt: The full prompt to continue from.

        Returns:
            Generated text chunk.
        """
        if self.llm is None:
            return ""
            
        if self.tokenizer is None:
             raise ValueError("Tokenizer required for generation")

        # Encode prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(self.llm, "device"):
             inputs = inputs.to(self.llm.device)

        # Generate
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                stop_strings=[self.config.call_scallop_token] if hasattr(self.tokenizer, "stop_strings") else None,
                # Note: stop_strings is newer transformers feature, fallback might be needed
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            
        # Decode
        # Skip prompt in output
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        
        return text

    def reset_state(self) -> None:
        """Reset the agent's session state."""
        self._state = AgentState()

    @property
    def history(self) -> list[ChatMessage]:
        """Get the conversation history."""
        return self._state.history.copy()
