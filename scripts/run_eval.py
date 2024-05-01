import wandb
from wandb.sdk.wandb_run import Run
import os
import sys
import argparse
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from config_singleton import WandbConfigSingleton
from cleanup import cleanup_gpu


def parse_args():
    parser = argparse.ArgumentParser(
        prog="nejumi eval", description="Run nejumi evaluation"
    )
    parser.add_argument("-c", "--config", default="configs/stable-base.yaml", help="Config file to use")
    parser.add_argument("-m", "--model", help="Model path or name to use")
    parser.add_argument("--llm-jp-eval", help="Run only llm-jp-eval", action="store_true")
    parser.add_argument("--mtbench", help="Run only mt-bench", action="store_true")

    args = parser.parse_args()
    return args


def load_config(conf_file, model_path):
    # Configuration loading
    if os.path.exists(conf_file):
        cfg = OmegaConf.load(conf_file)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(cfg_dict, dict)
        if model_path:
            # Note this assumes the tokenizer is included with the model.
            cfg_dict["model"]["pretrained_model_name_or_path"] = model_path
            cfg_dict["tokenizer"]["pretrained_model_name_or_path"] = model_path
            cfg_dict["wandb"]["run_name"] = f"nejumi eval {model_path}"
        # if the model path is actually an HF name, use that as the ID
        if not os.path.exists(model_path):
            cfg_dict["mtbench"]["model_id"] = model_path

    else:
        # Provide default settings in case config.yaml does not exist
        cfg_dict = {
            "wandb": {
                "entity": "default_entity",
                "project": "default_project",
                "run_name": "default_run_name",
            }
        }
    return cfg_dict


def wandb_setup(cfg_dict, conf_file):
    # W&B setup and artifact handling
    wandb.login()
    run = wandb.init(
        entity=cfg_dict["wandb"]["entity"],
        project=cfg_dict["wandb"]["project"],
        name=cfg_dict["wandb"]["run_name"],
        config=cfg_dict,
        job_type="evaluation",
    )

    # Initialize the WandbConfigSingleton
    table = wandb.Table(dataframe=pd.DataFrame())
    WandbConfigSingleton.initialize(run, table)
    cfg = WandbConfigSingleton.get_instance().config

    # add the run id to the model id
    # this allows multiple runs with the same model without collisions
    if cfg.mtbench.model_id is not None:
        cfg.mtbench.model_id = f"{cfg.mtbench.model_id}_{run.id}"

    # Save configuration as artifact
    if cfg.wandb.log:
        if os.path.exists(conf_file):
            artifact_config_path = conf_file
        else:
            # If the config file does not exist, write the contents of run.config as a YAML configuration string
            instance = WandbConfigSingleton.get_instance()
            assert isinstance(
                instance.config, DictConfig
            ), "instance.config must be a DictConfig"
            with open(conf_file, "w") as f:
                f.write(OmegaConf.to_yaml(instance.config))
            artifact_config_path = conf_file

        artifact = wandb.Artifact("config", type="config")
        artifact.add_file(artifact_config_path)
        run.log_artifact(artifact)

    return run, cfg, table


def run_llm_jp(run, config, table):
    from llm_jp_eval.evaluator import evaluate
    res = evaluate(run, config, table)
    cleanup_gpu()
    return res


def run_mt_bench(run, config, table):
    from mtbench_eval import mtbench_evaluate
    # If API keys are not set, fail fast
    try:
        os.environ["AZURE_OPENAI_KEY"]
    except KeyError:
        print("You need to load your OpenAI keys!")
        sys.exit(1)

    res = mtbench_evaluate(run, config, table)
    cleanup_gpu()
    return res


def finish(run, cfg, table):
    # Logging results to W&B
    if cfg.wandb.log and run is not None:
        run.log({"leaderboard_table": table})
        run.finish()


def main():
    args = parse_args()
    cfg_dict = load_config(args.config, args.model)
    run, cfg, table = wandb_setup(cfg_dict, args.config)

    # If only one is specified run that, otherwise run both
    if (not args.mtbench) and (not args.llm_jp_eval):
        table = run_llm_jp(run, cfg, table)
        table = run_mt_bench(run, cfg, table)
    elif not args.mtbench:
        run.name = run.name + " (llm-jp-eval only)"
        table = run_llm_jp(run, cfg, table)
    elif not args.llm_jp_eval:
        run.name = run.name + " (mtbench only)"
        table = run_mt_bench(run, cfg, table)

    finish(run, cfg, table)


if __name__ == "__main__":
    main()
