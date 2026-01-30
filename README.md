# 

This repository contains utilities and methods for weight-space merging and mergeability analysis. The repository is based on the implementation of **Donato Crisostomi** - [donatocrisostomi@gmail.com](mailto:donatocrisostomi@gmail.com)

---

## 🚀 Installation

You can install this project using [`uv`](https://github.com/astral-sh/uv):

```sh
uv sync
```

---


## Multi-Task Merging

Use `conf/multitask.yaml` to define the models you want to merge and the tasks you will evaluate the merged model on. Then run

```sh
uv run scripts/evaluate_multi_task_merging.py
```

If you want to define a new merging method, create a new class in `src/model_merging/merger/` and a corresponding config in `conf/merger`. Then change the `merger` field in the `multitask.yaml` config.
