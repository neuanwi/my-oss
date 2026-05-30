"""Run the full experiment: stream the dataset once per algorithm,
compute ground truth, measure accuracy/memory/time, sweep parameters,
and save figures + a JSON of results.
"""
import csv, time, sys, json, random
import tracemalloc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from algorithms import BloomFilter, CountMinSketch

DATA = "clickstream.csv"
random.seed(0)

# ---------- stream helper (one row at a time, never load whole file) ----------
def stream_rows():
    with open(DATA, newline="") as f:
        r = csv.reader(f)
        next(r)  # header
        for row in r:
            yield row  # [user_id, page_id, ts]

# ---------- Ground truth ----------
print("Computing ground truth...")
t0 = time.perf_counter()
true_page_freq = {}
true_users = set()
all_pages = set()
total = 0
for user_id, page_id, ts in stream_rows():
    true_page_freq[page_id] = true_page_freq.get(page_id, 0) + 1
    true_users.add(user_id)
    all_pages.add(page_id)
    total += 1
gt_time = time.perf_counter() - t0
print(f"  records={total}, unique_pages={len(all_pages)}, unique_users={len(true_users)}, gt_time={gt_time:.2f}s")

results = {"dataset": {"records": total, "unique_pages": len(all_pages),
                       "unique_users": len(true_users)}}

# ============================================================
# BLOOM FILTER experiment
# Task: membership test "has this user appeared?" — build on first 80% of
# distinct users, then test all users; measure false positive rate against
# the 20% held-out users that were never inserted.
# ============================================================
print("\n=== Bloom Filter ===")
user_list = sorted(true_users)
random.shuffle(user_list)
split = int(len(user_list) * 0.8)
inserted = set(user_list[:split])
heldout = user_list[split:]          # never inserted -> any positive is a false positive
n_insert = len(inserted)

def run_bloom(fp_target):
    m, k = BloomFilter.optimal_params(n_insert, fp_target)
    bf = BloomFilter(m, k)
    tracemalloc.start()
    t0 = time.perf_counter()
    for u in inserted:
        bf.add(u)
    build_time = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # measure observed FP rate on held-out users
    fp = sum(1 for u in heldout if u in bf)
    fp_rate = fp / len(heldout)
    # false negatives (should be zero by construction)
    fn = sum(1 for u in list(inserted)[:5000] if u not in bf)
    return {
        "fp_target": fp_target, "m_bits": m, "k": k,
        "struct_bytes": bf.size_bytes(),
        "peak_mem_bytes": peak,
        "build_time_s": build_time,
        "throughput": n_insert / build_time,
        "observed_fp_rate": fp_rate,
        "false_negatives": fn,
    }

bloom_sweep = [run_bloom(fp) for fp in [0.10, 0.05, 0.01, 0.001]]
for r in bloom_sweep:
    print(f"  fp_target={r['fp_target']:.3f}  m={r['m_bits']}  k={r['k']}  "
          f"size={r['struct_bytes']/1024:.1f}KB  observed_fp={r['observed_fp_rate']:.4f}  "
          f"FN={r['false_negatives']}  thr={r['throughput']:,.0f}/s")

# exact-set ground-truth memory baseline for comparison
tracemalloc.start()
exact_set = set(inserted)
_, exact_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
results["bloom"] = {"n_inserted": n_insert, "n_heldout": len(heldout),
                    "exact_set_bytes": exact_peak, "sweep": bloom_sweep}

# ============================================================
# COUNT-MIN SKETCH experiment
# Task: estimate per-page visit frequency from the stream.
# ============================================================
print("\n=== Count-Min Sketch ===")
def run_cms(eps, delta):
    w, d = CountMinSketch.optimal_params(eps, delta)
    cms = CountMinSketch(w, d)
    tracemalloc.start()
    t0 = time.perf_counter()
    for user_id, page_id, ts in stream_rows():
        cms.add(page_id)
    build_time = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # accuracy: relative error over all pages, and over top-100 heavy hitters
    errs = []
    for pg, tf in true_page_freq.items():
        est = cms.estimate(pg)
        errs.append(abs(est - tf) / tf)
    top = sorted(true_page_freq.items(), key=lambda x: -x[1])[:100]
    top_errs = [abs(cms.estimate(pg) - tf) / tf for pg, tf in top]
    mean_err = sum(errs) / len(errs)
    max_err = max(errs)
    top_err = sum(top_errs) / len(top_errs)
    return {
        "eps": eps, "delta": delta, "width": w, "depth": d,
        "struct_bytes": cms.size_bytes(),
        "peak_mem_bytes": peak,
        "build_time_s": build_time,
        "throughput": total / build_time,
        "mean_rel_err": mean_err,
        "max_rel_err": max_err,
        "top100_rel_err": top_err,
    }

cms_sweep = [run_cms(eps, 0.01) for eps in [0.01, 0.001, 0.0005, 0.0001]]
for r in cms_sweep:
    print(f"  eps={r['eps']:.4f}  w={r['width']}  d={r['depth']}  "
          f"size={r['struct_bytes']/1024:.1f}KB  mean_err={r['mean_rel_err']:.4f}  "
          f"top100_err={r['top100_rel_err']:.5f}  thr={r['throughput']:,.0f}/s")

# exact dict ground-truth memory baseline
tracemalloc.start()
exact_dict = dict(true_page_freq)
_, dict_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
results["cms"] = {"exact_dict_bytes": dict_peak, "sweep": cms_sweep}

# ---------- Save results ----------
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

# ============================================================
# FIGURES
# ============================================================
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})

# Fig 1: Bloom — observed FP vs target, and size vs FP target
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ft = [r["fp_target"] for r in bloom_sweep]
obs = [r["observed_fp_rate"] for r in bloom_sweep]
sz = [r["struct_bytes"] / 1024 for r in bloom_sweep]
ax[0].plot(ft, ft, "k--", label="target (ideal)")
ax[0].plot(ft, obs, "o-", color="#2c6fbb", label="observed")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("Target FP rate"); ax[0].set_ylabel("Observed FP rate")
ax[0].set_title("Bloom Filter: FP rate target vs observed"); ax[0].legend()
ax[1].plot(ft, sz, "s-", color="#c0504d")
ax[1].set_xscale("log")
ax[1].set_xlabel("Target FP rate"); ax[1].set_ylabel("Bit-array size (KB)")
ax[1].set_title("Bloom Filter: memory vs target FP rate")
fig.tight_layout(); fig.savefig("figs/bloom.png"); plt.close(fig)

# Fig 2: CMS — mean rel err vs size, and accuracy vs width
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
cs = [r["struct_bytes"] / 1024 for r in cms_sweep]
me = [r["mean_rel_err"] for r in cms_sweep]
te = [r["top100_rel_err"] for r in cms_sweep]
ax[0].plot(cs, me, "o-", color="#2c6fbb", label="all pages")
ax[0].plot(cs, te, "s-", color="#9bbb59", label="top-100 pages")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("Sketch size (KB)"); ax[0].set_ylabel("Mean relative error")
ax[0].set_title("Count-Min Sketch: accuracy vs memory"); ax[0].legend()
ws = [r["width"] for r in cms_sweep]
ax[1].plot(ws, me, "o-", color="#8064a2")
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("Width (w)"); ax[1].set_ylabel("Mean relative error")
ax[1].set_title("Count-Min Sketch: error vs width")
fig.tight_layout(); fig.savefig("figs/cms.png"); plt.close(fig)

# Fig 3: memory comparison vs exact baseline
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
bf_best = min(bloom_sweep, key=lambda r: abs(r["fp_target"] - 0.01))
ax[0].bar(["Exact set", "Bloom (1% FP)"],
          [results["bloom"]["exact_set_bytes"]/1024, bf_best["struct_bytes"]/1024],
          color=["#888", "#2c6fbb"])
ax[0].set_ylabel("Memory (KB)"); ax[0].set_title("Membership: exact vs Bloom")
cms_best = min(cms_sweep, key=lambda r: abs(r["eps"] - 0.001))
ax[1].bar(["Exact dict", "CMS (eps=1e-3)"],
          [results["cms"]["exact_dict_bytes"]/1024, cms_best["struct_bytes"]/1024],
          color=["#888", "#9bbb59"])
ax[1].set_ylabel("Memory (KB)"); ax[1].set_title("Frequency: exact vs CMS")
fig.tight_layout(); fig.savefig("figs/memory.png"); plt.close(fig)

print("\nSaved results.json and figures.")
