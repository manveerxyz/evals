import os
import re
import sys
import json
import argparse
import numpy as np

from pathlib import Path
from typing import Union
from print_color import print

from lm_eval import evaluator, utils, logger
from lm_eval.tasks import initialize_tasks, include_path
from lm_eval.api.registry import ALL_TASKS
from lm_eval.scholar_api import init_scholar_api, get_scholar_api
from lm_eval.config import get_config, APP_SHORT_NAME, APP_NAME, APP_VERSION
from bullet import Bullet, colors


def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "model",
        nargs="?",  # This makes the argument optional
        default=None,
        help="""
Model name from HuggingFace or OpenAI, or a path to a local model that can be loaded using `transformers.AutoConfig.from_pretrained`.
E.g.:
- HuggingFace Model: mistralai/Mistral-7B-v0.1
- OpenAI Model: gpt-3
- Local Model: ./path/to/model
        """.strip(),
    )
    parser.add_argument(
        "tasks",
        nargs="?",  # This makes the argument optional
        default=None,
        help="To get full list of tasks, use the command sevals --list-tasks",
    )
    parser.add_argument(
        "--model_args",
        default="",
        help="String arguments for model, e.g. 'dtype=float32'",
    )
    # TODO: List model args?
    parser.add_argument(
        "--gen_kwargs",
        default="",
        help=(
            "String arguments for model generation on greedy_until tasks,"
            " e.g. `temperature=0,top_k=0,top_p=0`"
        ),
    )

    parser.add_argument(
        "--list-tasks",
        type=str,
        metavar="[search string]",
        default=None,
        help="List all available tasks, that optionally match a search string, and exit.",
    )
    parser.add_argument(
        "--list-projects",
        action="store_true",
        help="List all projects you have on Scholar, and exit.",
    )

    parser.add_argument(
        "-p",
        "--project",
        default=None,
        help="ID of Scholar project to store runs/results in.",
    )

    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        help="Number of examples in few-shot context",
    )
    parser.add_argument("--batch_size", type=str, default=1)

    parser.add_argument(
        "-o",
        "--output_path",
        default="./out",
        type=str,
        metavar="[dir/file.jsonl] [DIR]",
        help="The path to the output file where the result metrics will be saved. If the path is a directory, the results will be saved in the directory. Else the parent directory will be used.",
    )
    parser.add_argument(
        "--include_path",
        type=str,
        default=None,
        help="Additional path to include if there are external tasks to include.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print verbose/detailed logs.",
    )
    return parser.parse_args()


def cli_evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    print(f"{APP_SHORT_NAME} ({APP_NAME})\nv{APP_VERSION}\n", color="blue")

    if not args:
        # we allow for args to be passed externally, else we parse them ourselves
        args = parse_eval_args()

    eval_logger = logger.eval_logger
    eval_logger.set_verbose(args.verbose)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    initialize_tasks()

    if args.include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")
        include_path(args.include_path)

    # list all projects
    if args.list_projects:
        config = get_config(require_api_key=True)
        init_scholar_api(config=config)
        api = get_scholar_api()
        valid, err_msg = api.check_token_expiry()
        if not valid:
            print(f"Unable to verify API Key: {err_msg}", color="yellow")
        projects, success = api.get_user_projects()
        if not success:
            print("Failed to fetch projects.", color="red")
            sys.exit()
        max_display_name_len = max(
            [len(project["display_name"]) for project in projects]
        )
        print(
            "Your Projects:\n - {}".format(
                "\n - ".join(
                    f"{project['display_name'].ljust(min(36, max_display_name_len))} ({project['slug']})"
                    for project in projects
                )
            )
        )
        sys.exit()
    if args.tasks is None or args.list_tasks:
        # if a search string was passed, filter tasks
        if args.list_tasks is not None and args.list_tasks != "":
            search_string = args.list_tasks
            matches = utils.fuzzy_match(search_string, ALL_TASKS)

            if not matches:
                print(
                    f"No tasks matching '{search_string}'. Try `sevals --list-tasks` for a list of available tasks.",
                    color="yellow",
                )
                exit(1)
            else:
                print(
                    "Tasks matching '{}':\n - {}".format(
                        search_string, "\n - ".join(sorted(matches))
                    )
                )
                sys.exit()

        print("All Available Tasks:\n - {}".format(f"\n - ".join(sorted(ALL_TASKS))))
        sys.exit()

    if args.model is None:
        sys.exit()

    if args.tasks is not None:
        if os.path.isdir(args.tasks):
            import glob

            task_names = []
            yaml_path = os.path.join(args.tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                config = utils.load_yaml_config(yaml_file)
                task_names.append(config)
        else:
            tasks_list = args.tasks.split(",")
            task_names = utils.pattern_match(tasks_list, ALL_TASKS)
            for task in [task for task in tasks_list if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)
            task_missing = [
                task
                for task in tasks_list
                if task not in task_names and "*" not in task
            ]  # we don't want errors if a wildcard ("*") task name was used

            if task_missing:
                missing = ", ".join(task_missing)
                eval_logger.error(f"Tasks were not found: {missing}\n")
                print(
                    f"Tasks {missing} were not found. Try `sevals --list-tasks` for a list of available tasks.",
                    color="yellow",
                )
                exit(1)

    if args.output_path:
        path = Path(args.output_path)
        # check if file or 'dir/results.json' exists
        if path.is_file() or Path(args.output_path).joinpath("results.json").is_file():
            filepath = path if path.is_file() else path.joinpath("results.json")
            print(
                f"Results file already exists at {filepath}. It will be overwritten.",
                color="yellow",
            )
            output_path_file = path.joinpath("results.json")
            assert not path.is_file(), "File already exists"
        # if path json then get parent dir
        elif path.suffix in (".json", ".jsonl"):
            output_path_file = path
            path.parent.mkdir(parents=True, exist_ok=True)
            path = path.parent
        else:
            path.mkdir(parents=True, exist_ok=True)
            output_path_file = path.joinpath("results.json")
    elif args.log_samples and not args.output_path:
        assert args.output_path, "Specify --output_path"

    # Get local config that contains the API Key
    config = get_config()
    init_scholar_api(config=config)
    api = get_scholar_api()

    eval_logger.info(f"Selected Tasks: {task_names}")

    project_id = args.project
    if args.project is None:
        print("Project (-p, --project) not specified.", color="yellow")
        # prompt user to select project
        projects, success = api.get_user_projects()
        if not success:
            print(
                "Failed to fetch projects, falling back to using 'default' project.",
                color="yellow",
            )
            project_id = "default"
        else:
            choices = [project["display_name"] for project in projects]
            choices.append("(Create a new project)")
            cli = Bullet(
                prompt="Select a project to save results to: ",
                choices=[choice for choice in choices],
                bullet="  -> ",  # Adding spaces for padding
                bullet_color=colors.foreground["white"],  # Bullet color
                word_color=colors.foreground["white"],  # Text color
                word_on_switch=colors.foreground["white"],  # Text color when selected
                background_color=colors.background["black"],  # Background color
                background_on_switch=colors.background[
                    "black"
                ],  # Background color when selected
                pad_right=5,  # Additional padding
            )
            project_name = cli.launch()
            if project_name == "(Create a new project)":
                project_name = input("Enter a name for the new project: ")
                # set project_id to the slug of the new project,
                # it'll get automatically made on init_run
                project_id = project_name
                print(f"Using project: {project_name}")
            else:
                project_id = next(
                    project["slug"]
                    for project in projects
                    if project["display_name"] == project_name
                )
                print(f"Using project: {project_name} ({project_id})")

    eval_logger.info(f"Recording results in project: {project_id}")

    run_id, err_msg = api.init_run(project_id=project_id)
    if err_msg is not None:
        print(f"Failed to initialize run: {err_msg}", color="red")
        sys.exit()

    print(f"Initialized run: {run_id}\n")

    results = evaluator.simple_evaluate(
        model_name=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        gen_kwargs=args.gen_kwargs,
    )

    if results is not None:
        samples = results.pop("samples")
        dumped = json.dumps(results, indent=2, default=_handle_non_serializable)

        scholar_results_url = api.record_final_report(
            versions=results["versions"],
            results=results["results"],
            configs=results["configs"],
            config=results["config"],
        )

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

        if args.output_path:
            output_path_file.open("w").write(dumped)

            for task_name, config in results["configs"].items():
                output_name = "{}_{}".format(
                    re.sub("/|=", "__", args.model_args), task_name
                )
                filename = path.joinpath(f"{output_name}.jsonl")
                samples_dumped = json.dumps(
                    samples[task_name], indent=2, default=_handle_non_serializable
                )
                filename.open("w").write(samples_dumped)

            print(f"Results saved to {output_path_file}", color="blue")

        print()
        print(
            f"{args.model} ({args.model_args}), gen_kwargs: ({args.gen_kwargs}), num_fewshot: {args.num_fewshot}, "
            f"batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
        )
        print(utils.make_table(results))
        if "groups" in results:
            print(utils.make_table(results, "groups"))

        if scholar_results_url:
            print(f"View results at:")
            print(f"{scholar_results_url}", color="green")


if __name__ == "__main__":
    cli_evaluate()
