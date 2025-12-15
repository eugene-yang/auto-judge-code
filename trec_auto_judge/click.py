from pathlib import Path
from .io import load_runs_failsave

def option_rag_responses():
    import click
    class ClickRagResponses(click.ParamType):
        name = "dir"

        def convert(self, value, param, ctx):
            if not value or not Path(value).is_dir():
                self.fail(f"The directory {value} does not exist, so I can not load rag responses from this directory.", param, ctx)
            runs = load_runs_failsave(Path(value))

            if len(runs) > 0:
                return runs

            self.fail(f"{value!r} contains no rag runs.", param, ctx)

    """Rag Run directory click option."""
    def decorator(func):
        func = click.option(
            "--rag-responses",
            type=ClickRagResponses(),
            required=True,
            help="The directory that contains the rag responses to evaluate."
        )(func)

        return func

    return decorator

def option_ir_dataset():
    import click
    from tira.third_party_integrations import ir_datasets
    from tira.ir_datasets_util import load_ir_dataset_from_local_file
    from ir_datasets import registry

    def irds_from_dir(directory):
        ds = load_ir_dataset_from_local_file(Path(directory), str(directory))
        if str(directory) not in registry:
            registry.register(str(directory), ds)
        return ds


    class ClickIrDataset(click.ParamType):
        name = "dir"

        def convert(self, value, param, ctx):
            if value == "infer-dataset-from-context":
                from huggingface_hub import DatasetCard
                candidate_files = set()
                if "rag_responses" in ctx.params:
                    for r in ctx.params["rag_responses"]:
                        if "path" in r:
                            p = Path(r["path"]).parent
                            candidate_files.add(p / "README.md")
                            candidate_files.add(p.parent / "README.md")
                irds_config = None
                base_path = None
                for c in candidate_files:
                    if c.is_file():
                        try:
                            irds_config = DatasetCard.load(str(c)).data["ir_dataset"]
                            base_path = c.parent
                            break
                        except:
                            pass
                if not irds_config:
                    raise ValueError("ToDo: Better error handling of wrong configurations")

                if "ir_datasets_id" in irds_config:
                    return ir_datasets.load(irds_config["ir_datasets_id"])
                elif "directory" in irds_config:
                    return irds_from_dir(str(base_path / irds_config["directory"]))
                else:
                    raise ValueError("ToDo: Better error handling of incomplete configurations")
                

            if value and value in registry:
                return ir_datasets.load(value)

            if value and Path(value).is_dir() and (Path(value) / "queries.jsonl").is_file() and (Path(value) / "corpus.jsonl.gz").is_file():
                return irds_from_dir(value)

            return ir_datasets.load(value)

    """Ir-dataset click option."""
    def decorator(func):
        func = click.option(
            "--ir-dataset",
            type=ClickIrDataset(),
            required=False,
            default="infer-dataset-from-context",
            help="The ir-datasets ID or a directory that contains the ir-dataset or TODO...."
        )(func)

        return func

    return decorator