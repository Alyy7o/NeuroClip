"""
verify_env.py
=============
Run this in a Kaggle notebook cell BEFORE starting the backend server.
It checks all required secrets/env vars and tells you exactly what is
missing and why.

Usage (in a Kaggle notebook cell):
    %run /kaggle/working/NeuroClip/backend/Semantic-search-app/backend/verify_env.py
"""

import os

REQUIRED = {
    "ASSEMBLYAI_API_KEY":      "AssemblyAI transcription (get at assemblyai.com)",
    "SUPABASE_URL":            "Supabase project URL",
    "SUPABASE_SERVICE_ROLE_KEY": "Supabase service-role key (Settings > API)",
}

OPTIONAL = {
    "GOOGLE_API_KEY":          "Google AI (used for LLM summaries)",
    "VITE_SUPABASE_ANON_KEY":  "Supabase anon key (for frontend)",
}

print("=" * 60)
print("  NeuroClip Environment Verification")
print("=" * 60)

# ── 1. Kaggle Secrets ──────────────────────────────────────────
print("\n[1] Kaggle Secrets:")
kaggle_ok = []
kaggle_fail = []
try:
    from kaggle_secrets import UserSecretsClient
    client = UserSecretsClient()
    all_keys = list(REQUIRED.keys()) + list(OPTIONAL.keys())
    for key in all_keys:
        try:
            val = client.get_secret(key)
            if val:
                kaggle_ok.append(key)
                print(f"    ✓  {key}  (len={len(val)})")
            else:
                kaggle_fail.append(key)
                print(f"    ✗  {key}  — secret exists but is EMPTY")
        except Exception as e:
            kaggle_fail.append(key)
            print(f"    ✗  {key}  — {e}")
except ImportError:
    print("    [not on Kaggle — kaggle_secrets not available]")
except Exception as e:
    print(f"    ERROR initialising UserSecretsClient: {e}")

# ── 2. Current os.environ ─────────────────────────────────────
print("\n[2] Current environment variables (os.environ):")
for key in list(REQUIRED.keys()) + list(OPTIONAL.keys()):
    val = os.environ.get(key, "")
    if val:
        print(f"    ✓  {key}  (len={len(val)})")
    else:
        print(f"    ✗  {key}  — NOT SET")

# ── 3. .env files ─────────────────────────────────────────────
print("\n[3] .env files found on disk:")
from pathlib import Path
candidates = [
    Path("/kaggle/working/NeuroClip/backend/Semantic-search-app/.env"),
    Path("/kaggle/working/NeuroClip/backend/.env"),
    Path("/kaggle/working/.env"),
]
found_any = False
for p in candidates:
    if p.exists():
        found_any = True
        print(f"    ✓  {p}  ({p.stat().st_size} bytes)")
        # Show which keys are in it
        for line in p.read_text(errors="replace").splitlines():
            s = line.strip()
            if s and not s.startswith("#") and "=" in s:
                k = s.split("=", 1)[0].strip()
                print(f"         key: {k}")
    else:
        print(f"    ✗  {p}  — not found")

# ── 4. Verdict ────────────────────────────────────────────────
print("\n" + "=" * 60)
missing = [k for k in REQUIRED if not os.environ.get(k)]
if missing:
    print("  ❌  MISSING required vars — backend WILL fail:")
    for k in missing:
        print(f"       {k}  ({REQUIRED[k]})")
    print("\n  Fix options:")
    print("   A) Add to Kaggle Secrets (Notebook sidebar → Add-ons → Secrets)")
    print("      then RESTART the kernel so secrets become available.")
    print()
    print("   B) Create /kaggle/working/NeuroClip/backend/Semantic-search-app/.env")
    print("      with the missing keys and restart.")
else:
    print("  ✅  All required vars are SET — backend should work.")
print("=" * 60)
