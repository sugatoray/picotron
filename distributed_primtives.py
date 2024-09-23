import os
import parallel_context as pc
import torch, torch.distributed as dist
import parallel_context as pc

STEP, VERBOSE = 0, os.environ.get("VERBOSE", "0") == "1"

def communicate(operation='send_forward', tensor=None, shapes=None, dtype=None):
    global STEP
    global VERBOSE
    if operation == 'recv_forward':
        if pc.parallel_context.pp_is_first_stage: return None
        tensor = torch.empty(shapes, requires_grad=True, device='cuda', dtype=dtype)
        src = pc.parallel_context.pp_prev_rank
    elif operation == 'send_forward':
        if pc.parallel_context.pp_is_last_stage: return
        dest = pc.parallel_context.pp_next_rank
    elif operation == 'recv_backward':
        if pc.parallel_context.pp_is_last_stage: return None
        tensor = torch.empty(shapes, requires_grad=True, device='cuda', dtype=dtype)
        src = pc.parallel_context.pp_next_rank
    elif operation == 'send_backward':
        if pc.parallel_context.pp_is_first_stage: return
        dest = pc.parallel_context.pp_prev_rank
    is_send = operation.startswith('send')
    peer_rank = dest if is_send else src
    op = dist.P2POp(dist.isend if is_send else dist.irecv, tensor, peer_rank)
    if VERBOSE: print(f"{operation} | {'sending' if is_send else 'receiving'} {operation.split('_')[1]} {pc.parallel_context.pp_rank} {'→' if is_send else '←'} {peer_rank} | STEP:{STEP} | RANK:{pc.parallel_context.pp_rank}", flush=True)
    [req.wait() for req in dist.batch_isend_irecv([op])]
    torch.cuda.synchronize()
    if VERBOSE: STEP += 1
    return tensor if not is_send else None

def bidirectional_communicate(operation, send_tensor, recv_shapes, dtype, device):
    global STEP
    global VERBOSE
    is_fwd = (operation == 'send_fwd_recv_bwd')
    if (is_fwd and pc.parallel_context.pp_is_last_stage) or (not is_fwd and pc.parallel_context.pp_is_first_stage): return None
    peer_rank = pc.parallel_context.pp_next_rank if is_fwd else pc.parallel_context.pp_prev_rank
    recv_tensor = torch.empty(recv_shapes, requires_grad=True, device=device, dtype=dtype)
    reqs = dist.batch_isend_irecv([dist.P2POp(dist.isend, send_tensor, peer_rank), dist.P2POp(dist.irecv, recv_tensor, peer_rank)])
    if VERBOSE: print(f"{operation} | sending {'next' if is_fwd else 'prev'} {pc.parallel_context.pp_rank} -> {peer_rank} | "f"receiving {'next' if is_fwd else 'prev'} {peer_rank} -> {pc.parallel_context.pp_rank} | "f"STEP {STEP=} | RANK:{pc.parallel_context.pp_rank}", flush=True)
    [req.wait() for req in reqs]
    torch.cuda.synchronize()
    if VERBOSE: STEP += 1
    return recv_tensor