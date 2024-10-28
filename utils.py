import torch
import random
import os
import numpy as np
import builtins
import fcntl
import src.distributed.process_group_manager as pgm

def print(*args, **kwargs):
    """ solves multi-process interleaved print problem """
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            builtins.print(*args, **kwargs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

def set_all_seed(seed):
    for module in [random, np.random]: module.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    
def to_readable_format(num, precision=2):
    if num >= 1e12:
        return f"{num / 1e12:.{precision}f}T"
    elif num >= 1e9:
        return f"{num / 1e9:.{precision}f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.{precision}f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"

def save_checkpoint(model, optimizer, trained_steps, trained_tokens, out_dir):
    """Save the model/optimizer states/steps to a checkpoint file."""
    tp_rank, pp_rank = pgm.process_group_manager.tp_rank, pgm.process_group_manager.pp_rank
    tp_world_size, pp_world_size = pgm.process_group_manager.tp_world_size, pgm.process_group_manager.pp_world_size
    ckpt_name = f"weights_tp_rank_world_size={tp_rank}_{tp_world_size}_pp_rank_world_size={pp_rank}_{pp_world_size}.pth"
    path = os.path.join(out_dir, ckpt_name)
    
    # Only DP/CP rank 0 will save the model, the weights are the same across all ranks
    if pgm.process_group_manager.dp_rank == 0 and pgm.process_group_manager.cp_rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        raw_model = model.module if pgm.process_group_manager.dp_world_size > 1 else model
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'trained_steps': trained_steps,
            'trained_tokens': trained_tokens
        }
        torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, out_dir):
    """Load the model/optimizer states from the latest checkpoint. Assume the topology is the same."""
    tp_rank, pp_rank = pgm.process_group_manager.tp_rank, pgm.process_group_manager.pp_rank
    tp_world_size, pp_world_size = pgm.process_group_manager.tp_world_size, pgm.process_group_manager.pp_world_size
    ckpt_name = f"weights_tp_rank_world_size={tp_rank}_{tp_world_size}_pp_rank_world_size={pp_rank}_{pp_world_size}.pth"
    path = os.path.join(out_dir, ckpt_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    checkpoint = torch.load(path)

    # Load model weights
    raw_model = model.module if pgm.process_group_manager.dp_world_size > 1 else model
    raw_model.load_state_dict(checkpoint['model'])
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['trained_steps'], checkpoint['trained_tokens']         
            
## def display_4D_parallelism_grid():
#    #TODO(fmom): fix me
#    #TODO(fmom): add color to distinguish between different parallelism groups
#    def create_gpu_box(gpu_num, tp, cp, pp):
#        return [
#            f"+------+",
#            f"|GPU:{gpu_num:<2d}|",
#            f"| TP:{tp:d} |",
#            f"| CP:{cp:d} |",
#            f"| PP:{pp:d} |",
#            f"+------+"
#        ]
#
#    def create_node(start_gpu, tp_size, cp_size, pp_size, node_index):
#        boxes = []
#        for i in range(8):  # 8 GPUs per node
#            gpu_num = start_gpu + i
#            tp = gpu_num % tp_size
#            cp = (gpu_num // tp_size) % cp_size
#            pp = (gpu_num // (tp_size * cp_size)) % pp_size
#            boxes.append(create_gpu_box(gpu_num, tp, cp, pp))
#        return ['  '.join(row) for row in zip(*boxes)]
#
#    def create_dp_box(replica_output):
#        width = len(replica_output[0]) + 4
#        top_bottom = f"+{'-' * (width - 2)}+"
#        return [top_bottom] + [f"| {line} |" for line in replica_output] + [top_bottom]
#
#    tp_size = pgm.process_group_manager.tp_size
#    cp_size = pgm.process_group_manager.cp_size
#    pp_size = pgm.process_group_manager.pp_size
#    dp_size = pgm.process_group_manager.dp_size
#    total_gpus_per_replica = tp_size * cp_size * pp_size
#    num_nodes_per_replica = (total_gpus_per_replica + 7) // 8  # Round up to nearest whole node
#
#    output = []
#    output.append("=== Simplified Parallelism Configuration ===")
#    output.append(f"TP Size: {tp_size}, CP Size: {cp_size}, PP Size: {pp_size}, DP Size: {dp_size}")
#    output.append(f"Total GPUs for one replica: {total_gpus_per_replica}")
#    output.append(f"Number of nodes per replica: {num_nodes_per_replica} (8 GPUs per node)")
#    output.append(f"Total GPUs: {total_gpus_per_replica * dp_size}")
#    output.append(f"Total nodes: {num_nodes_per_replica * dp_size}")
#    output.append("")
#
#    for dp in range(dp_size):
#        replica_output = []
#        for node in range(num_nodes_per_replica):
#            start_gpu = (dp * total_gpus_per_replica) + (node * 8)
#            node_output = create_node(start_gpu, tp_size, cp_size, pp_size, node)
#            replica_output.append(f"Node {dp * num_nodes_per_replica + node}:")
#            replica_output.extend(node_output)
#            replica_output.append("")
#
#        dp_box = create_dp_box(replica_output)
#        output.append(f"Data Parallel Group {dp}:")
#        output.extend(dp_box)
#        output.append("")
#
#    print("\n".join(output))
