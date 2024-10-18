import os
import src.distributed.process_group_manager as pgm
from typing import List, Optional
import torch, torch.distributed as dist

STEP, VERBOSE = 0, os.environ.get("VERBOSE", "0") == "1"

def pipeline_communicate(operation, device, dtype, tensor=None, shapes=None):
    global STEP
    global VERBOSE
    if operation == 'recv_forward':
        if pgm.process_group_manager.pp_is_first_stage: return None
        tensor = torch.empty(shapes, requires_grad=True, device=device, dtype=dtype)
        src = pgm.process_group_manager.pp_prev_rank
    elif operation == 'send_forward':
        if pgm.process_group_manager.pp_is_last_stage: return
        dest = pgm.process_group_manager.pp_next_rank
    elif operation == 'recv_backward':
        if pgm.process_group_manager.pp_is_last_stage: return None
        tensor = torch.empty(shapes, requires_grad=True, device=device, dtype=dtype)
        src = pgm.process_group_manager.pp_next_rank
    elif operation == 'send_backward':
        if pgm.process_group_manager.pp_is_first_stage: return
        dest = pgm.process_group_manager.pp_prev_rank
    is_send = operation.startswith('send')
    peer_rank = dest if is_send else src
    op = dist.P2POp(dist.isend if is_send else dist.irecv, tensor, peer_rank)
    if VERBOSE: print(f"{operation} | {'sending' if is_send else 'receiving'} {operation.split('_')[1]} {pgm.process_group_manager.pp_rank} {'→' if is_send else '←'} {peer_rank} | STEP:{STEP} | RANK:{pgm.process_group_manager.pp_rank}", flush=True)
    [req.wait() for req in dist.batch_isend_irecv([op])]
    torch.cuda.synchronize()
    if VERBOSE: STEP += 1
    return tensor if not is_send else None

def bidirectional_pipeline_communicate(operation, send_tensor, recv_shapes, device, dtype):
    global STEP
    global VERBOSE
    is_fwd = (operation == 'send_fwd_recv_bwd')
    if (is_fwd and pgm.process_group_manager.pp_is_last_stage) or (not is_fwd and pgm.process_group_manager.pp_is_first_stage): return None
    peer_rank = pgm.process_group_manager.pp_next_rank if is_fwd else pgm.process_group_manager.pp_prev_rank
    recv_tensor = torch.empty(recv_shapes, requires_grad=True, device=device, dtype=dtype)
    reqs = dist.batch_isend_irecv([dist.P2POp(dist.isend, send_tensor, peer_rank), dist.P2POp(dist.irecv, recv_tensor, peer_rank)])
    if VERBOSE: print(f"{operation} | sending {'next' if is_fwd else 'prev'} {pgm.process_group_manager.pp_rank} -> {peer_rank} | "f"receiving {'next' if is_fwd else 'prev'} {peer_rank} -> {pgm.process_group_manager.pp_rank} | "f"STEP {STEP=} | RANK:{pgm.process_group_manager.pp_rank}", flush=True)
    [req.wait() for req in reqs]
    torch.cuda.synchronize()
    if VERBOSE: STEP += 1
    return recv_tensor

class ContextComms:
    def __init__(self, msg: str = ""):
        global STEP
        global VERBOSE
        self._pending_operations: List[dist.P2POp] = []
        self._active_requests = None
        self.rank = pgm.process_group_manager.cp_rank
        self.world_size = pgm.process_group_manager.cp_world_size
        self.send_rank = pgm.process_group_manager.cp_send_rank
        self.recv_rank = pgm.process_group_manager.cp_recv_rank
        if VERBOSE: print(f"RingComm ({msg}) | initialized | RANK:{self.rank} | "f"WORLD_SIZE:{self.world_size} | SEND_RANK:{self.send_rank} | "f"RECV_RANK:{self.recv_rank}", flush=True)

    def send_recv(self, tensor_to_send, recv_tensor=None):
        if recv_tensor is None:
            result_tensor = torch.zeros_like(tensor_to_send)
        else:
            result_tensor = recv_tensor

        send_operation = dist.P2POp(dist.isend, tensor_to_send, self.send_rank, group=pgm.process_group_manager.cp_group)
        recv_operation = dist.P2POp(dist.irecv, result_tensor, self.recv_rank, group=pgm.process_group_manager.cp_group)
        
        self._pending_operations.extend([send_operation, recv_operation])

        if VERBOSE:
            print(f"RingComm | send_recv | STEP:{STEP} | RANK:{self.rank} | "f"ACTION:sending | TO:{self.send_rank} | TENSOR:{tensor_to_send}", flush=True)
            print(f"RingComm | send_recv | STEP:{STEP} | RANK:{self.rank} | "f"ACTION:receiving | FROM:{self.recv_rank} | TENSOR:{result_tensor}", flush=True)
        return result_tensor

    def commit(self):
        if self._active_requests is not None: raise RuntimeError("Commit called twice")
        self._active_requests = dist.batch_isend_irecv(self._pending_operations)
        if VERBOSE: print(f"RingComm | commit | STEP:{STEP} | RANK:{self.rank} | "f"ACTION:committed | NUM_OPS:{len(self._pending_operations) // 2}", flush=True)

    def wait(self):
        if self._active_requests is None: raise RuntimeError("Wait called before commit")
        for i, request in enumerate(self._active_requests):
            request.wait()
            if VERBOSE:
                operation_type = "send" if i % 2 == 0 else "receive"
                peer_rank = self.send_rank if operation_type == "send" else self.recv_rank
                print(f"RingComm | wait | STEP:{STEP} | RANK:{self.rank} | "f"ACTION:completed_{operation_type} | "f"{'FROM' if operation_type == 'receive' else 'TO'}:{peer_rank}", flush=True)
        torch.cuda.synchronize()
        self._active_requests = None
        self._pending_operations = []
        if VERBOSE: print(f"RingComm | wait | STEP:{STEP} | RANK:{self.rank} | "f"ACTION:all_operations_completed", flush=True)

def all_reduce_loss_across_dp_ranks(loss, device):
    reduced_loss = torch.tensor([loss if loss is not None else 0.0], dtype=torch.float32, device=device)
    # Reduce the loss across all workers so that every rank has the updated loss value.
    dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.world_group)
    reduced_loss /= pgm.process_group_manager.dp_world_size
    return reduced_loss.item()

def all_reduce_gradients_across_dp_cp_ranks(model):
    for param in model.parameters():
        if param.grad is not None:
            # Average the gradients across all DP & CP ranks
            param.grad /= pgm.process_group_manager.cp_dp_world_size
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.cp_dp_group)