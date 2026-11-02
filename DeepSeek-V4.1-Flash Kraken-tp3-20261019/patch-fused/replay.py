"""Offline replay of one BRRR fused decoder layer at TP3 (3 processes, one per GPU) from tensors dumped by
decoder.py (DS41_FUSED_DUMP). Meant to run under compute-sanitizer to locate the faulting kernel line.
  python replay.py --rank R --world 3 --dump /dump --layer 0 --port 29511
"""
import argparse, time
import torch, torch.distributed as dist

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True); ap.add_argument("--world", type=int, default=3)
ap.add_argument("--dump", default="/dump"); ap.add_argument("--layer", type=int, default=0)
ap.add_argument("--port", type=int, default=29511); ap.add_argument("--repeat", type=int, default=1)
a = ap.parse_args()
torch.cuda.set_device(a.rank)
dev = torch.device("cuda", a.rank)
dist.init_process_group("gloo", rank=a.rank, world_size=a.world, init_method=f"tcp://127.0.0.1:{a.port}")
from brrr import cuext  # noqa: E402

ext = cuext.extension("decoder")
print(f"[r{a.rank}] ext RANKS={ext.RANKS} OA={ext.OA} INTER={ext.INTER} RP_LAYOUT={ext.RP_LAYOUT}", flush=True)
handle = ext.create(cuext.exchange_ipc(ext, ext.flag_bytes() + ext.inbox_bytes(), a.rank, dist.group.WORLD), a.rank)
dist.barrier()
kw = torch.load(f"{a.dump}/layer{a.layer}_rank{a.rank}.pt", map_location=dev)
print(f"[r{a.rank}] layer tensors:", {k: (tuple(v.shape), str(v.dtype)) if isinstance(v, torch.Tensor) else v for k, v in kw.items()}, flush=True)
slot = ext.add_layer(handle, **kw)
rd = torch.load(f"{a.dump}/run{a.layer}_rank{a.rank}.pt", map_location=dev)
m, args = rd["m"], rd["args"]
print(f"[r{a.rank}] m={m} run args:", [(tuple(t.shape), str(t.dtype)) if isinstance(t, torch.Tensor) else t for t in args], flush=True)
HID, HC = ext.HID, ext.HC
f32 = torch.float32
for it in range(a.repeat):
    outs = (torch.empty(m, HID, dtype=torch.bfloat16, device=dev), torch.empty_like(args[4]),
            torch.empty(m, HC, dtype=f32, device=dev), torch.empty(m, HC, HC, dtype=f32, device=dev), torch.empty(m, HC, dtype=f32, device=dev))
    dist.barrier()
    t0 = time.time()
    ext.run(handle, slot, *args, *outs)
    torch.cuda.synchronize()
    print(f"[r{a.rank}] run {it} OK in {time.time()-t0:.2f}s; |out| sums:", [round(o.float().abs().sum().item(), 3) for o in outs],
          "nan:", [bool(o.isnan().any().item()) for o in outs], flush=True)
dist.barrier()
if a.rank == 0:
    torch.save([o.cpu() for o in outs], f"{a.dump}/replay_out_rank0.pt")
dist.destroy_process_group()
