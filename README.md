# Scholar Evals (sevals)

This is built on [Eleuther AI's LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) but has:
1. A simpler command-line interface
2. A UI to visualize results and view model outputs ([view example](https://usescholar.org/runs/cb249e62-a99f-468d-8eb2-b804fe31579a/results))

<img width="1440" alt="Screenshot 2023-12-20 at 7 48 32 PM" src="https://github.com/scholar-org/scholar-evals/assets/16143968/34763d52-f61e-4944-b941-65ae0cc6d9f1">

<img width="1440" alt="Screenshot 2023-12-20 at 7 49 12 PM" src="https://github.com/scholar-org/scholar-evals/assets/16143968/1423cc83-abc9-4239-90d1-3c2b77ab2f33">

## Installation

```bash
pip install sevals
```

### API Keys

Go to [usescholar.org/api-keys](https://usescholar.org/api-keys) to get an API Key, then enter it into the `sevals` CLI when prompted.

## Usage

```bash
sevals <model> <task> [options]
```

### Examples

Mock/Dummy model:
```bash
sevals dummy gsm8k
```

Local model:
```bash
sevals ./path/to/model gsm8k
```

HuggingFace model:
```bash
sevals mistralai/Mistral-7B-v0.1 gsm8k
```

OpenAI API:
```bash
sevals gpt-3.5-turbo gsm8k
```

### Tasks

Full list of tasks:
```bash
sevals --list_tasks
```

### Documentation

```bash
% sevals --help
usage: sevals [-h] [--model_args MODEL_ARGS] [--gen_kwargs GEN_KWARGS] [--list_tasks [search string]] [--list_projects] [-p PROJECT] [--num_fewshot NUM_FEWSHOT] [--batch_size BATCH_SIZE]
              [-o [dir/file.jsonl] [DIR]] [--include_path INCLUDE_PATH] [--verbose]
              [model] [tasks]

positional arguments:
  model                 Model name from HuggingFace or OpenAI, or a path to a local model that can be loaded using `transformers.AutoConfig.from_pretrained`.
                        E.g.:
                        - HuggingFace Model: mistralai/Mistral-7B-v0.1
                        - OpenAI Model: gpt-3
                        - Local Model: ./path/to/model
  tasks                 To get full list of tasks, use the command sevals --list_tasks

optional arguments:
  -h, --help            show this help message and exit
  --model_args MODEL_ARGS
                        String arguments for model, e.g. 'dtype=float32'
  --gen_kwargs GEN_KWARGS
                        String arguments for model generation on greedy_until tasks, e.g. `temperature=0,top_k=0,top_p=0`
  --list_tasks [search string]
                        List all available tasks, that optionally match a search string, and exit.
  --list_projects       List all projects you have on Scholar, and exit.
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
