"""
AIME 2025: https://huggingface.co/datasets/opencompass/AIME2025
"""

import pathlib
import random
import re

import pandas

from . import report
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult

current_dir = pathlib.Path(__file__).parent
current_dir = pathlib.Path(__file__).parent  # gpt_oss/evals
dataset_dir = current_dir / "datasets"

AIME_TEMPLATE = """
{question}
Please reason step by step, and put your final answer within \\boxed{{}}.
"""


def format_aime_question(row):
    return AIME_TEMPLATE.format(question=row["question"])


def extract_boxed_text(text):
    pattern = r"boxed{(.*?)}|framebox{(.*?)}"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        for match in matches[::-1]:
            for group in match:
                if group != "":
                    return group.split(",")[-1].strip()
    pattern = r"\d+"  # get the last integer if no pattern found
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1]
    return ""


def normalize_number(s):
    match = re.match(r"\d+", s)  # match digits from the start
    if not match:
        return None
    return match.group(0)


class AIME25Eval(Eval):
    def __init__(
        self,
        n_repeats: int = 4,
        num_examples: int
        | None = None,  # restrict to a subset of the data for debugging
        n_threads: int = 1,
    ):
        df1 = pandas.read_json(dataset_dir / "aime2025-I.jsonl", lines=True)
        df2 = pandas.read_json(dataset_dir / "aime2025-II.jsonl", lines=True)
        examples = [row.to_dict() for _, row in df1.iterrows()] + [
            row.to_dict() for _, row in df2.iterrows()
        ]
        examples = [
            {
                "question": row["question"],
                "answer": normalize_number(row["answer"])
                if isinstance(row["answer"], str)
                else row["answer"],
            }
            for row in examples
        ]
        rng = random.Random(0)
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported for num_examples = None"
            examples = rng.sample(examples, num_examples)
        examples = examples * n_repeats
        examples = [
            example | {"permutation": rng.sample(range(4), 4)} for example in examples
        ]
        self.examples = examples  # [:3]
        self.n_repeats = n_repeats
        self.n_threads = n_threads

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            fmt = format_aime_question(row)
            test_id = sampler.hash_prompt(fmt)
            print(f"Running test id aime25_{test_id}")
            prompt_messages = [sampler._pack_message(content=fmt, role="user")]
            sampler_response = sampler(prompt_messages)
            response_text = sampler_response.response_text
            actual_queried_prompt_messages = (
                sampler_response.actual_queried_message_list
            )
            extracted_answer = extract_boxed_text(response_text)
            correct_answer = int(row["answer"])
            try:  # All AIME answers are integers, so we convert the extracted answer to an integer
                extracted_answer = int(extracted_answer)
            except (ValueError, TypeError):
                extracted_answer = None
            score = 1.0 if extracted_answer == correct_answer else 0.0
            html = report.jinja_env.from_string(report.HTML_JINJA).render(
                prompt_messages=actual_queried_prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=correct_answer,
                extracted_answer=extracted_answer,
            )
            convo = actual_queried_prompt_messages + [
                dict(content=response_text, role="assistant")
            ]
            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics={"chars": len(response_text)},
                example_level_metadata={
                    "test_id": f"aime25_{test_id}",
                    "n_input_tokens": sampler_response.n_input_tokens,
                    "n_reasoning_tokens": sampler_response.n_reasoning_tokens,
                    "n_response_tokens": sampler_response.n_response_tokens,
                    "n_output_tokens": sampler_response.n_output_tokens,
                    "n_tool_calls_browser": sampler_response.n_tool_calls_browser,
                    "n_tool_calls_python": sampler_response.n_tool_calls_python,
                    "n_errors": sampler_response.n_errors,
                    "latency": sampler_response.latency,
                    "correct": score,
                },
            )

        results = report.map_with_progress(
            fn, self.examples, num_threads=self.n_threads
        )
        return report.aggregate_results(results)
