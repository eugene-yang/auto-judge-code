#!/usr/bin/env python3
import os
import click
import asyncio
from argparse import Namespace
from pathlib import Path




# TODO need to expose more hardcoded env variables from auto argue
# DEFAULT_MAX_CONCURRENCY, DEFAULT_REPORT_MAX_CHARS

# DEFAULT_REPORT_MAX_CHARS need to be set based on report requests

@click.command("auto-argue")
@click.option("--nugget_dir", type=Path, help="The nugget directory.", required=True)
@click.option("--collection", type=Path, help="Path to the collection file.", required=True)
@click.option("--output", type=Path, help="The output file.", required=True)
@click.option("--api_base_url", type=str, help="The API base URL.", default=None)
@click.option("--model_name", type=str, help="The model name to use.", default=None)
@click.option("--rerun_judgments", is_flag=True, help="Whether to rerun judgments.", default=False)
@click.option("--rag-responses", type=Path, help="The RAG responses to evaluate.", required=True)
def main(rag_responses: Path, rerun_judgments: bool, output: Path, model_name: str, api_base_url: str, collection: Path, nugget_dir: Path):
    output.parent.mkdir(exist_ok=True, parents=True)

    if api_base_url is None:
        api_base_url = os.environ.get('BASE_URL', os.environ['OPENAI_BASE_URL'])

    if model_name is None:
        model_name = os.environ.get(
            'MODEL_NAME',
            os.environ.get('OPENAI_MODEL_NAME', "meta-llama/Llama-3.3-70B-Instruct")
        )

    judgment_file = f"{output}.judgments.jsonl"
    raw_score_file = f"{output}.raw_score.tsv"
    os.environ['BASE_URL'] = api_base_url
    os.environ['MODEL_NAME'] = model_name

    if 'OPENAI_API_KEY' not in os.environ:
        os.environ['OPENAI_API_KEY'] = 'na'

    try:
        from auto_argue.eval import judge_run, score
    except ImportError:
        # we changed the name at some point...
        from argue_eval.cli import judge_run
        from argue_eval.score import score

    if not Path(judgment_file).exists() or rerun_judgments:
        asyncio.run(judge_run(Namespace(
            input_file=rag_responses,
            nuggets_file=nugget_dir,
            output_file_prefix=str(output),
            model_provider="from_env",
            model_name=model_name,
            collection=[collection],
            collection_dir="",
            topics=None,
            prompt_config_file=None,
            do_nugget_alignment_check=True,
            truncate_reports=True,
            always_check_all_nuggets=True,
            verbose=False,
            disable_cache=False
        ), judgment_file))


    score(
        judgments_jsonl=judgment_file,
        nuggets_path=nugget_dir,
        output_tsv=raw_score_file,
        topics=[],
        run_ids=[],
        validate=False,
        penalize_missing_topics=True,
    )

    with open(raw_score_file, "r") as fin, open(output, "w") as fout:
        _ = fin.readline() # skip header
        for line in fin:
            run_id, topic_id, metric, val = line.strip().split("\t")
            if '_micro' in metric:
                continue
            if '_macro' in metric:
                metric = metric.replace('_macro', '')
            fout.write(f"{run_id} {metric} {topic_id} {val}\n")

if __name__ == '__main__':
    main()
