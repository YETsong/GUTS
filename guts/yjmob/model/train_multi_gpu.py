"""Train GraphSAGE + GRU for next-grid prediction with multi-GPU data parallel.

Launch with torchrun; every rank runs the shared loop of
guts/yjmob/model/trainer.py on its own share of each day's sessions
(DayBatchSampler gives every rank the same number of same-day batches, padding
with empty batches instead of duplicating sessions). The model is not wrapped in
DistributedDataParallel: gradients are combined by one token-weighted all-reduce
per step (``DDPContext.sync_gradients``). Rank 0 writes the run
directory, checkpoints and results. guts/yjmob/pipeline.py does not use this
file; it trains shards with train_single.py.

Example:
    torchrun --nproc_per_node=4 guts/yjmob/model/train_multi_gpu.py --processed_dir <dir> --epochs 100
"""

from __future__ import annotations

import os
import sys
import warnings
from datetime import timedelta
from typing import List

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning)
import torch
import torch.distributed as dist
import torch.nn as nn

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from guts.yjmob.model.trainer import SingleProcess, build_arg_parser, run_training


class DDPContext(SingleProcess):
    """Process context of one torchrun rank."""

    def __init__(self) -> None:
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        timeout_min = int(os.environ.get("DDP_TIMEOUT_MIN", "120"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_min))
        super().__init__(device)

    def sync_gradients(self, model: nn.Module, n_tokens: int) -> None:
        """Token-weighted average of this step's gradients over ranks (one flat all-reduce).

        Each rank's gradient is of its own mean loss over n_tokens tokens, so
        sum_r(n_r * g_r) / sum_r(n_r) equals the gradient of the mean loss over all
        ranks' tokens, as if the step's batches were one batch. A rank without
        gradients (filler batch, skipped non-finite loss) joins with n_tokens = 0.
        If no rank has tokens, gradients are cleared so the optimizer step does
        nothing, as in a single-process run.
        """
        params = list(model.parameters())
        n = float(n_tokens)
        flat = torch.cat([
            (p.grad * n if (p.grad is not None and n > 0) else torch.zeros_like(p)).flatten() for p in params
        ] + [torch.full((1,), n, device=self.device)])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        total = float(flat[-1].item())
        if total == 0:
            for p in params:
                p.grad = None
            return
        flat.div_(total)
        offset = 0
        for p in params:
            numel = p.numel()
            p.grad = flat[offset:offset + numel].view_as(p).clone()
            offset += numel

    def sum(self, values: List[float]) -> List[float]:
        t = torch.tensor([float(v) for v in values], dtype=torch.float64, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t.tolist()

    def broadcast(self, obj):
        holder = [obj]
        dist.broadcast_object_list(holder, src=0)
        return holder[0]

    def barrier(self) -> None:
        dist.barrier()

    def close(self) -> None:
        dist.destroy_process_group()


def main() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        raise SystemExit("train_multi_gpu.py must be launched with torchrun (WORLD_SIZE > 1); use train_single.py for one process.")
    args = build_arg_parser("GraphSAGE + GRU next-grid training (multi-GPU data parallel).").parse_args()
    ctx = DDPContext()
    try:
        run_training(args, ctx, log_label=ctx.rank)
    finally:
        ctx.close()


if __name__ == "__main__":
    main()
