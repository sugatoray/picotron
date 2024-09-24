import torch, random, numpy as np
import parallel_context as pc

def set_all_seed(seed):
    for module in [random, np.random]: module.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def display_parallelism_grid():
    def _create_box(content):
        return f"  {content:^3}  "

    def _create_row(row):
        return "|" + "|".join(_create_box(f"g{num:02d}") for num in row) + "|"

    def _create_border(width):
        return "+" + "-" * (width - 2) + "+"

    def _create_pp_line(width, pp_size):
        box_width = (width - pp_size + 1) // pp_size
        return "  ".join("PP".center(box_width) for _ in range(pp_size))

    output = []
    sample_row = _create_row(pc.parallel_context.grid[0, :, 0])
    row_width = len(sample_row)
    border = _create_border(row_width)

    output.append(f"=== Global Parallelism Configuration ===")
    output.append(f"DP Size: {pc.parallel_context.dp_size}, PP Size: {pc.parallel_context.pp_size}, TP Size: {pc.parallel_context.grid.shape[0]}")
    output.append("")  # Top spacing

    for dp in range(pc.parallel_context.dp_size):
        output.append(f"DP {dp}:")
        output.append(f"{'':>8}{border}")
        
        for tp in range(pc.parallel_context.grid.shape[0]):
            if tp == 0:
                output.append(f"{'TP':>7} {_create_row(pc.parallel_context.grid[tp, :, dp])}")
            else:
                output.append(f"{'':8}{border}")
                output.append(f"{'TP':>7} {_create_row(pc.parallel_context.grid[tp, :, dp])}")
        
        output.append(f"{'':8}{border}")
        if pc.parallel_context.pp_size > 1:
            output.append(f"{'':>7}{_create_pp_line(row_width, pc.parallel_context.pp_size)}")

        output.append("")  # Spacing between DP blocks

    output.append("")  # Bottom spacing

    output.append(f"=== Local Parallelism Configuration ===")
    output.append(pc.parallel_context.__str__())            
    output.append(f"DP Group IDs: {['g{:02d}'.format(id) for id in pc.parallel_context.dp_group_ids]}")
    output.append(f"PP Group IDs: {['g{:02d}'.format(id) for id in pc.parallel_context.pp_group_ids]}")
    output.append(f"TP Group IDs: {['g{:02d}'.format(id) for id in pc.parallel_context.tp_group_ids]}")

    print("\n".join(output))
