"""3-process unit test of the RANKS=3 phased extension: gather_words, all_gather, all_reduce vs torch references."""
import os, sys, torch, torch.distributed as dist
rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank); dev = torch.device("cuda", rank)
dist.init_process_group("gloo", rank=rank, world_size=world)
from brrr.phased_all_reduce import PhasedAllReduce
par = PhasedAllReduce(dist.group.WORLD, dev)
assert not par.disabled, "disabled: world != RANKS"
bad = []
def check(name, ok):
    if not ok: bad.append(name)
g = torch.Generator(device="cpu").manual_seed(1234)
# 1) gather_words: (rows, 5) fp32 words -> (ranks, rows, 5); 60 rounds to exercise parity/epoch
for it in range(60):
    rows = [1, 3, 8, 13, 48][it % 5]
    inp = (torch.randn(rows, 5, generator=g) * 100 + rank * 1000).to(dev)
    out = par.gather_words(inp)
    torch.cuda.synchronize(dev)
    ref = [torch.empty_like(inp.cpu()) for _ in range(world)]
    dist.all_gather(ref, inp.cpu())
    ref = torch.stack(ref).to(dev)
    check(f"gather_words it={it} rows={rows}", out.shape == ref.shape and torch.equal(out, ref))
# 2) all_gather along last dim: the logits shapes (rows, 43136) fp32 and a few others
for it in range(40):
    rows = [1, 2, 5, 6, 8][it % 5]
    inp = torch.randn(rows, 43136, generator=g).to(dev) + rank
    if not par.should_gather(inp, -1):
        check(f"all_gather eligible rows={rows}", False); continue
    out = par.all_gather(inp, -1)
    torch.cuda.synchronize(dev)
    ref = [torch.empty_like(inp.cpu()) for _ in range(world)]
    dist.all_gather(ref, inp.cpu())
    ref = torch.cat(ref, dim=-1).to(dev)
    check(f"all_gather it={it} rows={rows}", out.shape == ref.shape and torch.equal(out, ref))
# 3) all_reduce bf16: (40,5120) (48,5120) (56,6144) (2,6144)
for it in range(40):
    shape = [(40, 5120), (48, 5120), (56, 6144), (2, 6144), (8, 5120)][it % 5]
    inp = (torch.randn(*shape, generator=g) + rank).to(torch.bfloat16).to(dev)
    if not par.should_use(inp):
        check(f"all_reduce eligible {shape}", False); continue
    out = par.all_reduce(inp)
    torch.cuda.synchronize(dev)
    ref = inp.float().cpu().clone(); dist.all_reduce(ref)
    err = (out.float().cpu() - ref).abs().max().item()
    check(f"all_reduce it={it} {shape} maxerr={err:.4f}", err <= 0.06 * (1 + ref.abs().max().item()) )
dist.barrier()
print(f"[rank {rank}] phased RANKS=3 test: {'ALL OK' if not bad else 'FAILURES: ' + '; '.join(bad[:8]) + (' ...' if len(bad) > 8 else '')}", flush=True)
dist.destroy_process_group()
sys.exit(1 if bad else 0)
