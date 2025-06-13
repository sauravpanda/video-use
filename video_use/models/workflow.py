"""Workflow-related data models."""

from typing import Optional, Dict
from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """Token usage tracking for LLM calls."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model_name: str = ""
    call_count: int = 0
    
    def add_usage(self, input_tokens: int, output_tokens: int, model_name: str = ""):
        """Add token usage from a single LLM call."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += (input_tokens + output_tokens)
        self.call_count += 1
        if model_name and not self.model_name:
            self.model_name = model_name
    
    def merge(self, other: 'TokenUsage'):
        """Merge token usage from another TokenUsage instance."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.call_count += other.call_count


class StructuredWorkflowOutput(BaseModel):
    """Structured output format for LLM-generated workflow instructions."""
    prompt: str = Field(
        description="Human-readable workflow instructions with numbered steps"
    )
    start_url: str = Field(
        description="The starting URL where the workflow should begin execution"
    )
    parameters: Dict[str, str] = Field(
        default_factory=dict,
        description="Dictionary of parameters containing data used for actions"
    )
    token_usage: Optional[TokenUsage] = Field(
        default=None,
        description="Token usage information for the LLM call"
    ) 