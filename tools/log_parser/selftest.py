import parser
import collections
for f in ["example/screen_example.log", "example/screen_example2.log"]:
    rows = parser.parse_log(open(f, encoding="utf-8", errors="replace").read())
    miss = [r for r in rows if r["s2_time"] is not None and r["s2_fft"] is None]
    same = [r for r in rows if r["s2_fft"] is not None and r["s2_fft"] == r["s1_fft"]]
    no_s1 = [r for r in rows if r["s1_fft"] is None]
    types = set((r["s1_fft_type"], r["s2_fft_type"]) for r in rows if r["s1_fft_type"])
    print(f)
    print("  runs", len(rows),
          "| status", dict(collections.Counter(r["status"] for r in rows)))
    print("  complete-missing-s2fft", len(miss),
          "| s1==s2 count", len(same),
          "| runs-without-s1fft", len(no_s1))
    print("  fft type pairs:", types)

