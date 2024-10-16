import torch
import random
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
