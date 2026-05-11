import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, overload

Message = dict[str, Any]  # keys role, content
MessageList = list[Message]


@dataclass
class SamplerResponse:
    """
    Response from a sampler.
    """

    response_text: str
    actual_queried_message_list: MessageList
    response_metadata: dict[str, Any]

    n_input_tokens: int = 0
    n_reasoning_tokens: int = 0
    n_response_tokens: int = 0
    n_output_tokens: int = 0

    n_tool_calls_browser: int = 0
    n_tool_calls_python: int = 0
    n_errors: int = 0

    latency: float = 0.0


class SamplerBase:
    """
    Base class for defining a sampling model, which can be evaluated,
    or used as part of the grading process.
    """

    def __call__(
        self,
        message_list: MessageList,
    ) -> SamplerResponse:
        raise NotImplementedError

    def hash_prompt(self, prompt: str) -> str:
        hash_bytes = hashlib.sha256(prompt.encode()).digest()
        return str(uuid.UUID(bytes=hash_bytes[:16]))


@dataclass
class EvalResult:
    """
    Result of running an evaluation (usually consisting of many samples)
    """

    score: float | None  # top-line metric
    metrics: dict[str, float] | None  # other metrics
    htmls: list[str]  # strings of valid HTML
    convos: list[MessageList]  # sampled conversations
    metadata: dict[str, Any] | None  # Extra data such as rubric scores or sollen


@dataclass
class SingleEvalResult:
    """
    Result of evaluating a single sample
    """

    score: float | None
    metrics: dict[str, float] = field(default_factory=dict)
    html: str | None = None
    convo: MessageList | None = None  # sampled conversation
    example_level_metadata: dict[str, Any] | None = (
        None  # Extra data such as rubric scores or sollen
    )


class Eval:
    """
    Base class for defining an evaluation.
    """

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        raise NotImplementedError
