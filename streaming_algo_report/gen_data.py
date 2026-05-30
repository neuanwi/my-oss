"""Generate a synthetic clickstream dataset (>=100,000 records).
Each line: user_id,page_id,timestamp
Distribution is Zipf-like so a few pages dominate (realistic for frequency sketches).
"""
import random, csv

random.seed(42)
N = 200_000
N_USERS = 25_000
N_PAGES = 8_000

# Zipf-ish page popularity
pages = list(range(N_PAGES))
weights = [1.0 / (i + 1) ** 0.9 for i in range(N_PAGES)]

with open("clickstream.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["user_id", "page_id", "ts"])
    ts = 1_700_000_000
    for _ in range(N):
        u = random.randint(1, N_USERS)
        p = random.choices(pages, weights=weights, k=1)[0]
        ts += random.randint(0, 3)
        w.writerow([f"u{u}", f"p{p}", ts])

print(f"wrote clickstream.csv with {N} records")
