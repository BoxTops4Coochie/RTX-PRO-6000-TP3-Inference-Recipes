"""Replay layer L: staged dense copies vs the kernel reading vLLM's row-major weights (dense_native=True)."""
import argparse, time
import torch, torch.distributed as dist
ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True); ap.add_argument("--layer", type=int, default=0)
ap.add_argument("--dump", default="/dump"); ap.add_argument("--port", type=int, default=29541)
a = ap.parse_args()
torch.cuda.set_device(a.rank); dev = torch.device("cuda", a.rank)
dist.init_process_group("gloo", rank=a.rank, world_size=3, init_method=f"tcp://127.0.0.1:{a.port}")
from brrr import cuext
ext = cuext.extension("decoder")
handle = ext.create(cuext.exchange_ipc(ext, ext.flag_bytes() + ext.inbox_bytes(), a.rank, dist.group.WORLD), a.rank)
dist.barrier()
kw = torch.load(f"{a.dump}/layer{a.layer}_rank{a.rank}.pt", map_location=dev)
nat = torch.load(f"{a.dump}/native{a.layer}_rank{a.rank}.pt", map_location=dev)
csf = {k: v for k, v in kw.items() if k.startswith("csf")}
base = {k: v for k, v in kw.items() if not k.startswith("csf")}
slot_s = ext.add_layer(handle, **base, **csf)
kn = dict(base); kn.update({k: nat[k].contiguous() for k in ("woa", "wob", "gate", "sh13", "sh2")})
slot_n = ext.add_layer(handle, **kn, **csf, dense_native=True)
rd = torch.load(f"{a.dump}/run{a.layer}_rank{a.rank}.pt", map_location=dev); m, args = rd["m"], rd["args"]
HID, HC, f32 = ext.HID, ext.HC, torch.float32
def run(slot):
    outs = (torch.empty(m, HID, dtype=torch.bfloat16, device=dev), torch.empty_like(args[4]),
            torch.empty(m, HC, dtype=f32, device=dev), torch.empty(m, HC, HC, dtype=f32, device=dev), torch.empty(m, HC, dtype=f32, device=dev))
    dist.barrier(); t0 = time.time(); ext.run(handle, slot, *args, *outs); torch.cuda.synchronize(); return outs, time.time() - t0
def cmp(tag, x, y):
    rel = [float((p.float() - q.float()).abs().max() / (q.float().abs().max() + 1e-9)) for p, q in zip(x, y)]
    print(f"[r{a.rank}] {tag}: rel={[f'{d:.2g}' for d in rel]} nan={[bool(p.isnan().any()) for p in x]}", flush=True)
s1, t1 = run(slot_s); s2, _ = run(slot_s); n1, t3 = run(slot_n); n2, _ = run(slot_n)
for _ in range(20): run(slot_s)
ts = sum(run(slot_s)[1] for _ in range(20)) / 20; tn = sum(run(slot_n)[1] for _ in range(20)) / 20
print(f"[r{a.rank}] m={m} avg ms staged {ts*1e3:.3f} native {tn*1e3:.3f}", flush=True)
cmp("staged vs staged", s1, s2); cmp("native vs native", n1, n2); cmp("staged vs native", s1, n1)
dist.barrier(); dist.destroy_process_group()
