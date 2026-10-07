"""Train GraphSAGE + GRU for next-grid prediction in a single process (one GPU or CPU).

guts/yjmob/pipeline.py and the unlearning retrain run this once per shard, one
subprocess per GPU (CUDA_VISIBLE_DEVICES selects the GPU). The training loop is
guts/yjmob/model/trainer.py; for multi-GPU data parallel use train_multi_gpu.py.

Example:
    python guts/yjmob/model/train_single.py --processed_dir <shard dir> --epochs 100
"""

from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning)
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from guts.yjmob.model.trainer import SingleProcess, build_arg_parser, run_training


def main() -> None:
    args = build_arg_parser("GraphSAGE + GRU next-grid training (single process).").parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Shards train in parallel, one process per GPU: label logs by the physical GPU and
    # let only the process on GPU 0 echo to stdout.
    log_label = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    run_training(args, SingleProcess(device), log_label=log_label)


if __name__ == "__main__":
    main()
