#!/usr/bin/env python3
from trec_auto_judge.click import option_rag_responses, option_rag_topics
import click
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from statistics import mean
import random

from trec_auto_judge.request import Request

# crucible
from nuggety.align import evaluator_run, Umbrela
from nuggety.alignment_result import *
import dspy
from typing import *
from pydantic import BaseModel





class UmbrelaAnnotation(BaseModel):
    query_id:str
    title_query:str
    source_document:str
    problem_statement:str=""
    background:str=""
    confidence:Optional[float] = None
    reasoning:Optional[str] = None
    answerability:Optional[str] = None
    is_match:Optional[bool] = None
    match_score:Optional[float] = None

@click.command("umbrela_baseline")
@click.option("--output", type=Path, help="The output file.", required=True)
@option_rag_responses()
@option_rag_topics()

def main(rag_responses: list[dict], rag_topics: List[Request], output: Path):
    """
    A naive rag response assessor that just orders each response by its length.
    """
    ret = []
    avg_grades = defaultdict(list)
    avg_ismatch = defaultdict(list)
    topic_dict = {request.request_id: request for request in rag_topics}
    alignment_input_list = list()

    for rag_response in rag_responses:
        metadata = rag_response["metadata"]
        run_id = metadata["run_id"]
        topic_id = metadata["narrative_id"]
        
        text = " ".join([i["text"] for i in rag_response["answer"]])
        
        topic = topic_dict[topic_id]
        if topic is None:
            raise RuntimeError("Could not identify request object for topic {topic_id}")
        
        alignment_inputs = UmbrelaAnnotation(query_id = topic_id
                                            ,source_document = text
                                            ,metadata= metadata
                                            ,title_query = topic.title if topic.title else ""
                                            ,background = topic.background if topic.background else ""
                                            ,problem_statement =  topic.problem_statement if topic.problem_statement else ""
                                            )
        alignment_input_list.append(alignment_inputs)


    alignment_results =  evaluator_run(prompt=Umbrela, output_converter=Umbrela.convert_output,alignment_input_list=alignment_input_list)
       
    for res in alignment_results:
        ret.append(f"{run_id} GRADE {topic_id} {res.match_score}")
        ret.append(f"{run_id} IS_MATCH {topic_id} {res.is_match}")
        avg_grades[run_id].append(res.match_score)
        avg_ismatch[run_id].append(res.is_match)
        
    ret.append(f"{run_id} GRADES all {mean([float(g) for g in avg_grades[run_id]])}")
    ret.append(f"{run_id} ISMATCH all {mean([1.0 if b else 0.0 for b in avg_ismatch[run_id]])}")

    output.parent.mkdir(exist_ok=True, parents=True)
    output.write_text("\n".join(ret))


if __name__ == '__main__':
    main()