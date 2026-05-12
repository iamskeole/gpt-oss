import argparse
import json
import os
from datetime import datetime

from . import report
from .aime_eval import AIME25Eval
from .basic_eval import BasicEval
from .chat_completions_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    ChatCompletionsSampler,
)
from .gpqa_eval import GPQAEval
from .healthbench_eval import HealthBenchEval
from .responses_sampler import ResponsesSampler

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "sk-none"


# -------------------------------------------------------------------
# Helper to parse boolean-like arguments that can accept variants
# such as true/false, 1/0, yes/no, etc.  It also supports the flag
# form –flag (equivalent to `--flag true`).
# -------------------------------------------------------------------
class BoolOption(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # `values` is None when the flag is provided without a value.
        if values is None:
            val = True
        else:
            v = values.lower()
            if v in ("true", "1", "t", "yes", "y"):
                val = True
            elif v in ("false", "0", "f", "no", "n"):
                val = False
            else:
                parser.error(
                    f"argument {option_string}: invalid boolean value '{values}'. "
                    "Expected true/false, 1/0, yes/no, t/f, y/n"
                )
        setattr(namespace, self.dest, val)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-20b",
        help="Select a model by name. Accepts a comma-separated list.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="medium",
        help="Reasoning effort (low, medium, high). Accepts a comma-separated list.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["responses", "chat"],
        default="responses",
        help="Sampler backend to use for models.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:9999/v1",
        help="Base URL for the API.",
    )
    parser.add_argument(
        "--eval",
        type=str,
        default="aime25",
        help="Select an eval by name. Accepts a comma-separated list.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=1,
        help="Number of threads to run.",
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--examples", type=int, help="Number of examples to use (overrides default)"
    )
    parser.add_argument(
        "--enable-browser-tool",
        action="store_true",
        default=False,
        help="Enable the builtin browser tool.",
    )
    parser.add_argument(
        "--enable-python-tool",
        action="store_true",
        default=False,
        help="Enable the builtin python tool.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/tmp/gpt_oss_eval",
        help="Location of results output.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=69421337,
        help="Seed for controling deterministic outputs.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=1,
        help="How many times to run the same benchmark",
    )

    args = parser.parse_args()

    sampler_cls = (
        ResponsesSampler if args.sampler == "responses" else ChatCompletionsSampler
    )

    models = {}
    for model_name in args.model.split(","):
        for reasoning_effort in args.reasoning_effort.split(","):
            models[f"{model_name}-{reasoning_effort}"] = sampler_cls(
                model=model_name,
                reasoning_model=True,
                reasoning_effort=reasoning_effort,
                temperature=args.temperature,
                base_url=args.base_url,
                max_tokens=131_072,
                enable_browser_tool=args.enable_browser_tool,
                enable_python_tool=args.enable_python_tool,
                seed=args.seed,
            )

    print(f"Running with args {args}")

    grading_sampler = ChatCompletionsSampler(
        model="gpt-4.1-2025-04-14",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
        base_url="https://api.openai.com/v1",
    )

    def get_evals(eval_name, debug_mode):
        num_examples = (
            args.examples if args.examples is not None else (5 if debug_mode else None)
        )
        # Set num_examples = None to reproduce full evals
        match eval_name:
            case "basic":
                return BasicEval()
            case "gpqa":
                return GPQAEval(
                    n_repeats=1 if args.debug else args.n_repeats,  # was hardcoded to 8
                    num_examples=num_examples,
                    debug=debug_mode,
                    n_threads=args.n_threads or 1,
                )
            case "healthbench":
                return HealthBenchEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                    n_repeats=1,
                    n_threads=args.n_threads or 1,
                    subset_name=None,
                )
            case "healthbench_hard":
                return HealthBenchEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                    n_repeats=1,
                    n_threads=args.n_threads or 1,
                    subset_name="hard",
                )
            case "healthbench_consensus":
                return HealthBenchEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                    n_repeats=1,
                    n_threads=args.n_threads or 1,
                    subset_name="consensus",
                )
            case "aime25":
                return AIME25Eval(
                    n_repeats=1 if args.debug else args.n_repeats,  # was hardcoded to 8
                    num_examples=num_examples,
                    n_threads=args.n_threads or 1,
                )
            case _:
                raise Exception(f"Unrecognized eval type: {eval_name}")

    evals = {}
    for eval_name in args.eval.split(","):
        evals[eval_name] = get_evals(eval_name, args.debug)

    debug_suffix = "_DEBUG" if args.debug else ""
    print(debug_suffix)
    mergekey2resultpath = {}
    print(f"Running the following evals: {evals}")
    print(f"Running evals for the following models: {models}")

    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M%S")
    out_root = args.results_dir
    os.makedirs(out_root, exist_ok=True)

    for model_name, sampler in models.items():
        model_name = model_name.replace("/", "__")
        for eval_name, eval_obj in evals.items():
            result = eval_obj(sampler)
            # ^^^ how to use a sampler
            file_stem = f"{eval_name}_{model_name}_temp{args.temperature}"
            # file stem should also include the year, month, day, and time in hours and minutes
            file_stem += f"_{date_str}"
            report_filename = f"{out_root}/{file_stem}{debug_suffix}.html"
            print(f"Writing report to {report_filename}")
            with open(report_filename, "w") as fh:
                fh.write(report.make_report(result))
            assert result.metrics is not None
            metrics = result.metrics | {"score": result.score}
            # Sort metrics by key
            metrics = dict(sorted(metrics.items()))
            print(metrics)
            result_filename = f"/{out_root}/{file_stem}{debug_suffix}.json"
            with open(result_filename, "w") as f:
                f.write(json.dumps(metrics, indent=2))
            print(f"Writing results to {result_filename}")

            full_result_filename = (
                f"{out_root}/{file_stem}{debug_suffix}_allresults.json"
            )
            with open(full_result_filename, "w") as f:
                result_dict = {
                    "score": result.score,
                    "metrics": result.metrics,
                    "htmls": result.htmls,
                    "convos": result.convos,
                    "metadata": result.metadata,
                }
                f.write(json.dumps(result_dict, indent=2))
                print(f"Writing all results to {full_result_filename}")

            mergekey2resultpath[f"{file_stem}"] = result_filename

    merge_metrics = []
    for eval_model_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue
        result = result.get("f1_score", result.get("score", None))
        eval_name = eval_model_name[: eval_model_name.find("_")]
        model_name = eval_model_name[eval_model_name.find("_") + 1 :]
        merge_metrics.append(
            {"eval_name": eval_name, "model_name": model_name, "metric": result}
        )
    print(merge_metrics)
    return merge_metrics


if __name__ == "__main__":
    main()
