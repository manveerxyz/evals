# Scholar Eval (sevals)

This is built on [Eleuther AI's LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) but has:
1. A simpler command-line interface
2. A UI to visualize results and view model outputs

<img width="1440" alt="Screenshot 2023-12-12 at 8 58 55 PM" src="https://github.com/scholar-org/scholar-evals/assets/16143968/e7612e96-3de7-408a-8b1c-fa1b6758d414">

<img width="1440" alt="Screenshot 2023-12-12 at 8 59 10 PM" src="https://github.com/scholar-org/scholar-evals/assets/16143968/6d262a2b-45d9-422f-bdf5-ff8cc8c60fe4">


## Installation

```bash
pip install sevals
```

## Usage

```bash
sevals <model> <task> [options]
```

### Examples

Mock/Dummy model:
```bash
sevals dummy lambada_openai
```

Local model:
```bash
sevals ./path/to/model lambada_openai
```

HuggingFace model:
```bash
sevals hf mistralai/Mistral-7B-v0.1 lambada_openai
```

OpenAI API:
```bash
sevals gpt-3.5-turbo lambada_openai
```

### Tasks

Full list of tasks:
```bash
sevals --list-tasks
```

### Documentation

```bash
% sevals --help
usage: sevals [-h] [--model_args MODEL_ARGS] [--gen_kwargs GEN_KWARGS] [--list-tasks [search string]] [--list-projects] [-p PROJECT] [--num_fewshot NUM_FEWSHOT] [--batch_size BATCH_SIZE]
              [-o [dir/file.jsonl] [DIR]] [--include_path INCLUDE_PATH] [--verbose]
              [model] [tasks]

positional arguments:
  model                 Model name from HuggingFace or OpenAI, or a path to a local model that can be loaded using `transformers.AutoConfig.from_pretrained`.
                        E.g.:
                        - HuggingFace Model: mistralai/Mistral-7B-v0.1
                        - OpenAI Model: gpt-3
                        - Local Model: ./path/to/model
  tasks                 To get full list of tasks, use the command sevals --list-tasks

optional arguments:
  -h, --help            show this help message and exit
  --model_args MODEL_ARGS
                        String arguments for model, e.g. 'dtype=float32'
  --gen_kwargs GEN_KWARGS
                        String arguments for model generation on greedy_until tasks, e.g. `temperature=0,top_k=0,top_p=0`
  --list-tasks [search string]
                        List all available tasks, that optionally match a search string, and exit.
  --list-projects       List all projects you have on Scholar, and exit.
  -p PROJECT, --project PROJECT
                        ID of Scholar project to store runs/results in.
  --num_fewshot NUM_FEWSHOT
                        Number of examples in few-shot context
  --batch_size BATCH_SIZE
  -o [dir/file.jsonl] [DIR], --output_path [dir/file.jsonl] [DIR]
                        The path to the output file where the result metrics will be saved. If the path is a directory, the results will be saved in the directory. Else the parent directory will be used.
  --include_path INCLUDE_PATH
                        Additional path to include if there are external tasks to include.
  --verbose             Whether to print verbose/detailed logs.
```
