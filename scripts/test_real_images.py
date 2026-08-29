"""Upload the 7 real example images to the running server and print a summary table."""
import httpx
import pathlib

BASE = "http://127.0.0.1:8001"

rows = []
for img in sorted(pathlib.Path("real_examples").glob("*.jpg")):
    with open(img, "rb") as f:
        r = httpx.post(f"{BASE}/api/v1/analyses", files={"file": (img.name, f, "image/jpeg")})
    if r.status_code != 201:
        print(f"FAILED {img.name}: {r.status_code} {r.text[:200]}")
        continue
    d = r.json()
    issues = ", ".join(i["type"] for i in d.get("issues", [])) or "none"
    rows.append((img.name, d.get("quality_label"), d.get("quality_score"), issues, d.get("timing_ms", {}).get("total", 0)))

header = f"{'File':<32} {'Label':<22} {'Score':>5}  {'Issues':<50}  {'ms':>6}"
print(header)
print("-" * len(header))
for name, label, score, issues, ms in rows:
    print(f"{name:<32} {label:<22} {score:>5.1f}  {issues:<50}  {ms:>6.0f}")
