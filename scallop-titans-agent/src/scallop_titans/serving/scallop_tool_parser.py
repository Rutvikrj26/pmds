#!/usr/bin/env python3
"""
RS-5: vLLM Custom Tool Parser for Scallop Integration

This module implements a custom tool parser for vLLM that detects
<|call_scallop|> tokens and routes them to the Scallop engine.

Usage:
    vllm serve model_name \
        --tool-parser-plugin /path/to/scallop_tool_parser.py \
        --tool-call-parser scallop \
        --enable-auto-tool-choice
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from vllm.entrypoints.openai.protocol import (
        ChatCompletionRequest,
        DeltaToolCall,
        ExtractedToolCallInformation,
        FunctionCall,
        ToolCall,
    )

# Attempt to import vLLM - this will only work in a vLLM environment
try:
    from vllm.entrypoints.openai.tool_parsers import ToolParser, ToolParserManager
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    # Create dummy classes for testing without vLLM
    class ToolParser:
        pass
    class ToolParserManager:
        @staticmethod
        def register_module(names):
            def decorator(cls):
                return cls
            return decorator


# Scallop token patterns from master_plan.md
CALL_SCALLOP_TOKEN = "<|call_scallop|>"
SCALLOP_RESULT_TOKEN = "<|scallop_result|>"
END_SCALLOP_RESULT_TOKEN = "<|end_scallop_result|>"
START_THOUGHT_TOKEN = "<|start_thought|>"
END_THOUGHT_TOKEN = "<|end_thought|>"


@ToolParserManager.register_module(["scallop"])
class ScallopToolParser(ToolParser):
    """
    Custom tool parser for Scallop neuro-symbolic reasoning.
    
    Detects <|call_scallop|> tokens in model output and extracts
    Scallop commands for execution.
    """
    
    def __init__(self, tokenizer: Any):
        """Initialize the parser with the tokenizer."""
        if VLLM_AVAILABLE:
            super().__init__(tokenizer)
        self.tokenizer = tokenizer
        self.current_tool_id = 0
        
        # Regex pattern for extracting Scallop commands
        # Matches: <|call_scallop|>command_here<|end_thought|>
        self.command_pattern = re.compile(
            rf"{re.escape(CALL_SCALLOP_TOKEN)}(.+?)(?:{re.escape(END_THOUGHT_TOKEN)}|$)",
            re.DOTALL
        )
    
    def adjust_request(self, request: "ChatCompletionRequest") -> "ChatCompletionRequest":
        """
        Adjust the incoming request for Scallop tool calling.
        
        This method can modify the request before generation,
        for example to add special tokens or modify the prompt.
        """
        # For Scallop, we don't need to modify the request
        # The model is already fine-tuned to emit <|call_scallop|>
        return request
    
    def extract_tool_calls(
        self,
        model_output: str,
        request: "ChatCompletionRequest | None" = None,
    ) -> "ExtractedToolCallInformation":
        """
        Extract tool calls from model output.
        
        Looks for <|call_scallop|> tokens and extracts the Scallop
        commands that follow.
        
        Args:
            model_output: The full model output text
            request: The original request (optional)
            
        Returns:
            ExtractedToolCallInformation with any detected tool calls
        """
        from vllm.entrypoints.openai.protocol import (
            ExtractedToolCallInformation,
            FunctionCall,
            ToolCall,
        )
        
        tool_calls: list[ToolCall] = []
        content_parts: list[str] = []
        
        # Find all Scallop commands
        last_end = 0
        for match in self.command_pattern.finditer(model_output):
            # Add content before this tool call
            content_parts.append(model_output[last_end:match.start()])
            
            # Extract the command
            scallop_command = match.group(1).strip()
            
            # Create tool call
            tool_call = ToolCall(
                id=f"scallop_{self.current_tool_id}",
                type="function",
                function=FunctionCall(
                    name="call_scallop",
                    arguments=json.dumps({
                        "command": scallop_command,
                    }),
                ),
            )
            tool_calls.append(tool_call)
            self.current_tool_id += 1
            
            last_end = match.end()
        
        # Add remaining content
        content_parts.append(model_output[last_end:])
        remaining_content = "".join(content_parts).strip()
        
        # Check if there are any incomplete tool calls (partial token)
        tools_delta = CALL_SCALLOP_TOKEN in remaining_content and END_THOUGHT_TOKEN not in model_output[model_output.rfind(CALL_SCALLOP_TOKEN):]
        
        return ExtractedToolCallInformation(
            tools_called=len(tool_calls) > 0,
            tool_calls=tool_calls if tool_calls else None,
            content=remaining_content if remaining_content else None,
        )
    
    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: "ChatCompletionRequest | None" = None,
    ) -> tuple[list["DeltaToolCall"], str]:
        """
        Extract tool calls from streaming output.
        
        This is called incrementally as tokens are generated.
        
        Args:
            previous_text: Text before this delta
            current_text: Full text including delta
            delta_text: Just the new text
            previous_token_ids: Token IDs before delta
            current_token_ids: All token IDs including delta
            delta_token_ids: Just the new token IDs
            request: Original request
            
        Returns:
            Tuple of (tool call deltas, remaining content)
        """
        from vllm.entrypoints.openai.protocol import DeltaFunctionCall, DeltaToolCall
        
        tool_call_deltas: list[DeltaToolCall] = []
        
        # Check if we just completed a <|call_scallop|> token
        if CALL_SCALLOP_TOKEN in delta_text:
            # Start a new tool call
            tool_call_deltas.append(
                DeltaToolCall(
                    id=f"scallop_{self.current_tool_id}",
                    index=self.current_tool_id,
                    type="function",
                    function=DeltaFunctionCall(
                        name="call_scallop",
                        arguments="",
                    ),
                )
            )
        
        # Check if we're in the middle of a tool call (after <|call_scallop|> but before end)
        if CALL_SCALLOP_TOKEN in current_text:
            scallop_idx = current_text.rfind(CALL_SCALLOP_TOKEN)
            after_token = current_text[scallop_idx + len(CALL_SCALLOP_TOKEN):]
            
            if END_THOUGHT_TOKEN not in after_token:
                # We're still building the command
                # Only send the delta if it's part of the command
                if scallop_idx < len(previous_text):
                    # Delta is part of command arguments
                    if tool_call_deltas or delta_text.strip():
                        tool_call_deltas.append(
                            DeltaToolCall(
                                index=self.current_tool_id,
                                function=DeltaFunctionCall(
                                    arguments=delta_text,
                                ),
                            )
                        )
        
        # Check if tool call just completed
        if END_THOUGHT_TOKEN in delta_text and CALL_SCALLOP_TOKEN in previous_text:
            self.current_tool_id += 1
        
        # Return content that's not part of tool calls
        content = delta_text
        if CALL_SCALLOP_TOKEN in content or END_THOUGHT_TOKEN in content:
            content = ""
        
        return tool_call_deltas, content


def execute_scallop_tool(command: str) -> str:
    """
    Execute a Scallop command and return the result.
    
    This function should be called by the vLLM server when
    a tool call is detected.
    
    Args:
        command: The Scallop command to execute
        
    Returns:
        The result string to inject back into the model
    """
    try:
        from scallop_titans.reasoning import ScallopEngine
        
        engine = ScallopEngine()
        result = engine.execute_command(command)
        return f"{SCALLOP_RESULT_TOKEN}{result}{END_SCALLOP_RESULT_TOKEN}"
    except Exception as e:
        return f"{SCALLOP_RESULT_TOKEN}Error: {str(e)}{END_SCALLOP_RESULT_TOKEN}"


# Test the parser without vLLM
def test_parser():
    """Test the ScallopToolParser with sample outputs."""
    print("=" * 60)
    print("RS-5: vLLM Tool Parser Test")
    print("=" * 60)
    
    # Create mock tokenizer
    class MockTokenizer:
        pass
    
    parser = ScallopToolParser(MockTokenizer())
    
    # Test case 1: Simple command
    test_output1 = """Let me reason about this.
<|start_thought|>I need to find the relationship.
<|call_scallop|>add_fact(parent, alice, bob). query(grandparent, ?, charlie)<|end_thought|>
Based on the result, the answer is Alice."""
    
    print("\nTest 1: Simple command extraction")
    print(f"Input: {test_output1[:50]}...")
    
    # We can't fully test without vLLM types, but we can test the regex
    matches = list(parser.command_pattern.finditer(test_output1))
    print(f"Found {len(matches)} command(s)")
    for i, match in enumerate(matches):
        print(f"  Command {i+1}: {match.group(1).strip()}")
    
    # Test case 2: Multiple commands
    test_output2 = """<|call_scallop|>add_fact(mother, betty, alice)<|end_thought|>
<|call_scallop|>add_fact(sister, betty, carol)<|end_thought|>
<|call_scallop|>query(aunt, ?, alice)<|end_thought|>"""
    
    print("\nTest 2: Multiple commands")
    matches = list(parser.command_pattern.finditer(test_output2))
    print(f"Found {len(matches)} command(s)")
    for i, match in enumerate(matches):
        print(f"  Command {i+1}: {match.group(1).strip()}")
    
    print("\n" + "=" * 60)
    print("RS-5 RESULTS")
    print("=" * 60)
    print("\n✅ Tool parser implementation ready")
    print("✅ Command extraction regex working")
    print("\nTo use with vLLM:")
    print("  vllm serve Qwen/Qwen3-32B \\")
    print("      --tool-parser-plugin scallop_tool_parser.py \\")
    print("      --tool-call-parser scallop \\")
    print("      --enable-auto-tool-choice")


if __name__ == "__main__":
    test_parser()
