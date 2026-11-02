"""Replay layer L (CSF layer, e.g. 20) twice on 3 GPUs: RAW path (s13/s2 = decoded scales from the dump) vs CSF path
(payload storage + gdesc built from the csf dump). Outputs must match bit-exactly.
  python replay_csf.py --rank R --layer 20 --dump /dump
"""
import argparse, time
import torch, torch.distributed as dist

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True); ap.add_argument("--world", type=int, default=3)
ap.add_argument("--dump", default="/dump"); ap.add_argument("--layer", type=int, default=20); ap.add_argument("--port", type=int, default=29521)
ap.add_argument("--repeat", type=int, default=1)
a = ap.parse_args()
torch.cuda.set_device(a.rank); dev = torch.device("cuda", a.rank)
dist.init_process_group("gloo", rank=a.rank, world_size=a.world, init_method=f"tcp://127.0.0.1:{a.port}")
from brrr import cuext  # noqa: E402
from brrr.decoder import Engine  # noqa: E402  (vllm import side effects acceptable in the image)

ext = cuext.extension("decoder")
handle = ext.create(cuext.exchange_ipc(ext, ext.flag_bytes() + ext.inbox_bytes(), a.rank, dist.group.WORLD), a.rank)
dist.barrier()
kw = torch.load(f"{a.dump}/layer{a.layer}_rank{a.rank}.pt", map_location=dev)
kw = {k: v for k, v in kw.items() if not k.startswith("csf")}  # newer dumps include the CSF kwargs; rebuilt below
info = torch.load(f"{a.dump}/csf{a.layer}_rank{a.rank}.pt", map_location=dev, weights_only=False)
assert info["csf"], "dump is not a CSF layer"
# sanity: the dumped s13/s2 must equal the decoded reference (dump_csf ran decode_all right before the kwargs dump)
for k, r in (("s13", "ref_w13_sfb"), ("s2", "ref_w2_sfb")):
    same = torch.equal(kw[k].view(torch.uint8).reshape(-1), info[r].to(dev).contiguous().view(torch.uint8).reshape(-1))
    print(f"[r{a.rank}] dumped {k} == reference: {same}", flush=True)


class Scales:  # minimal stand-in for CsfBlockPayload fields used by Engine.csf_args
    def __init__(self, d):
        self.storage = d["storage"].to(dev); self.rows, self.columns, self.num_experts = d["rows"], d["columns"], d["num_experts"]
        self.destination, self.native_codes, self.group_record_bytes = d["destination"], d["native_codes"], d["group_record_bytes"]
        self.quantization, self.sections = d["quantization"], d["sections"]


csf = {}
for tag, name in (("csf13", "first"), ("csf2", "second")):
    r = Engine.csf_args(Scales(info[name]))
    csf[f"{tag}_storage"], csf[f"{tag}_gdesc"], csf[f"{tag}_meta"] = r["storage"], r["gdesc"], r["meta"]
    print(f"[r{a.rank}] {tag}: meta={r['meta']} gdesc={tuple(r['gdesc'].shape)} max_count={int(((r['gdesc'] >> 24) & 0xff).max())}", flush=True)
# gdesc self-check against a direct numpy derivation of the group record bounds
import numpy as np
for tag, name in (("csf13", "first"), ("csf2", "second")):
    d = info[name]; st = d["storage"].cpu().numpy(); R, C, Es = d["rows"], d["columns"], d["num_experts"]
    meta = csf[f"{tag}_meta"]; bases, codes, code_bytes, value_off, recpos, R_, cg, groups, sections = meta
    pages = (groups + 31) // 32; prefixes = groups + pages
    page_bases = codes + Es * code_bytes; page_bases = (page_bases + 15) // 16 * 16
    offsets = (page_bases + Es * pages * 4 + 15) // 16 * 16
    pb = st[page_bases:page_bases + Es * pages * 4].view("<u4").reshape(Es, pages)
    off = st[offsets:offsets + Es * prefixes * 2].view("<u2").reshape(Es, prefixes)
    sg = np.arange(groups); pg = sg // 32
    first = pb[:, pg].astype(np.int64) + off[:, sg + pg]; cnt = pb[:, pg].astype(np.int64) + off[:, sg + pg + 1] - first
    g = csf[f"{tag}_gdesc"].cpu().numpy().view(np.uint32).reshape(Es, groups)
    print(f"[r{a.rank}] {tag} gdesc check: first ok={bool(((g & 0xffffff) == first).all())} count ok={bool(((g >> 24) == np.minimum(cnt, 255)).all())} recpos_ok={recpos == (offsets + Es * prefixes * 2 + 15) // 16 * 16}", flush=True)
dbg13 = torch.full((384 * 6 * 40 * 256,), -1, dtype=torch.int32, device=dev)
dbg2 = torch.full((384 * 20 * 6 * 256,), -1, dtype=torch.int32, device=dev)
slot_raw = ext.add_layer(handle, **kw)
slot_csf = ext.add_layer(handle, **kw, **csf, csf13_debug=dbg13, csf2_debug=dbg2)
rd = torch.load(f"{a.dump}/run{a.layer}_rank{a.rank}.pt", map_location=dev)
m, args = rd["m"], rd["args"]
HID, HC, f32 = ext.HID, ext.HC, torch.float32


def run(slot):
    outs = (torch.empty(m, HID, dtype=torch.bfloat16, device=dev), torch.empty_like(args[4]),
            torch.empty(m, HC, dtype=f32, device=dev), torch.empty(m, HC, HC, dtype=f32, device=dev), torch.empty(m, HC, dtype=f32, device=dev))
    dist.barrier()
    t0 = time.time(); ext.run(handle, slot, *args, *outs); torch.cuda.synchronize()
    return outs, time.time() - t0


def cmp(tag, x, y):
    eq = [torch.equal(p, q) for p, q in zip(x, y)]
    diff = [float((p.float() - q.float()).abs().max()) for p, q in zip(x, y)]
    rel = [float((p.float() - q.float()).abs().max() / (q.float().abs().max() + 1e-9)) for p, q in zip(x, y)]
    print(f"[r{a.rank}] {tag}: equal={eq} maxdiff={[f'{d:.3g}' for d in diff]} rel={[f'{d:.2g}' for d in rel]}", flush=True)


raw1, t1 = run(slot_raw); raw2, t2 = run(slot_raw); csf1, t3 = run(slot_csf); csf2, t4 = run(slot_csf)
print(f"[r{a.rank}] m={m} times ms: raw {t1*1e3:.2f} {t2*1e3:.2f} csf {t3*1e3:.2f} {t4*1e3:.2f}; |y| max {float(raw1[0].float().abs().max()):.3g} mean {float(raw1[0].float().abs().mean()):.3g}", flush=True)
cmp("raw vs raw", raw1, raw2); cmp("csf vs csf", csf1, csf2); cmp("raw vs csf", raw1, csf1)
# where do they differ? rows/cols count
d = (raw1[0].float() - csf1[0].float()).abs()
print(f"[r{a.rank}] y diff: elements differing {int((d > 0).sum())} / {d.numel()}, rows with any diff {int((d > 0).any(1).sum())} / {d.shape[0]}, >1e-1: {int((d > 0.1).sum())}", flush=True)
# compare the mirrored expanded words with the reference scales (only positions the kernel wrote)
for tag, dbg, ref, shape in (("w13", dbg13, info["ref_w13_sfb"], (384, 6, 40, 256)), ("w2", dbg2, info["ref_w2_sfb"], (384, 20, 6, 256))):
    got = dbg.cpu().numpy().view(np.uint32).reshape(shape)
    exp = ref.cpu().contiguous().view(torch.int32).numpy().view(np.uint32).reshape(shape)
    written = got != 0xFFFFFFFF
    bad = written & (got != exp)
    print(f"[r{a.rank}] {tag} expanded words: written={int(written.sum())} mismatching={int(bad.sum())}", flush=True)
    for idx in np.argwhere(bad)[:6]:
        e, nt256, kt, row = (int(x) for x in idx)
        print(f"[r{a.rank}]   e={e} nt256={nt256} kt={kt} row={row} got={got[e,nt256,kt,row]:08x} exp={exp[e,nt256,kt,row]:08x}", flush=True)
dist.barrier(); dist.destroy_process_group()
