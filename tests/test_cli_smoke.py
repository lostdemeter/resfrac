import os
import sys
import subprocess
import shlex

PROJECT_ROOT = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, os.pardir))


def run(cmd: str, timeout: int = 120):
    """Run a shell command and return (code, out, err)."""
    proc = subprocess.run(
        shlex.split(cmd),
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        text=True,
        env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_holo_index_demo_smoke():
    # Keep it tiny so CI stays fast
    cmd = (
        f"{shlex.quote(sys.executable)} -m resfrac.tools.holo_index_demo "
        "--N 2000 --d 4 --Q 10 --S 10 --K 64 "
        "--dtype float32 --chunk-k 32 --probes 1 --ensembles 1"
    )
    code, out, err = run(cmd, timeout=120)
    assert code == 0, f"holo_index_demo failed: {err}\n{out}"
    assert "Holographic Sublinear Index Demo" in out
    assert "top-1 recall:" in out


def test_holo_index_demo_fresnel_smoke():
    # Exercise Fresnel propagation path with gating on
    cmd = (
        f"{shlex.quote(sys.executable)} -m resfrac.tools.holo_index_demo "
        "--N 1000 --d 4 --Q 6 --S 8 --K 32 --prop fresnel --gate "
        "--dtype float32 --chunk-k 16 --probes 1 --ensembles 1"
    )
    code, out, err = run(cmd, timeout=120)
    assert code == 0, f"holo_index_demo (fresnel) failed: {err}\n{out}"
    assert "Holographic Sublinear Index Demo" in out
    assert "top-1 recall:" in out


def test_prime_cli_smoke():
    # Small N keeps it fast and deterministic
    cmd = (
        f"{shlex.quote(sys.executable)} -m resfrac.primes.cli "
        "--N 2000 --backend chudnovsky"
    )
    code, out, err = run(cmd, timeout=120)
    assert code == 0, f"prime CLI failed: {err}\n{out}"
    assert "backend" in out.lower() or "primes" in out.lower()


def test_bench_holo_smoke():
    # Minimal benchmark sweep; 1 trial, small sizes. Avoids long prime runs.
    cmd = (
        f"{shlex.quote(sys.executable)} bench_holo.py "
        "--trials 1 --tsp-n 10 --sat uf50-0218.cnf --primeN 2000 "
        "--prop angular --phase-retrieval gs "
        "--holoindex --hi-N 2000 --hi-d 4 --hi-Q 10 --hi-S 10 --hi-K 64 --hi-metric euclidean"
    )
    code, out, err = run(cmd, timeout=180)
    assert code == 0, f"bench_holo failed: {err}\n{out}"
    assert "== TSP ==" in out
    # The benchmark prints the section as '== 3-SAT =='
    assert "== 3-SAT ==" in out
    assert "== Primes ==" in out
    assert "== HoloIndex ==" in out

    # Run a second minimal bench to exercise hilbert phase retrieval path
    cmd2 = (
        f"{shlex.quote(sys.executable)} bench_holo.py "
        "--trials 1 --tsp-n 10 --sat uf50-0218.cnf --primeN 2000 "
        "--prop none --phase-retrieval hilbert "
        "--holoindex --hi-N 1000 --hi-d 4 --hi-Q 6 --hi-S 8 --hi-K 32 --hi-metric euclidean"
    )
    code2, out2, err2 = run(cmd2, timeout=180)
    assert code2 == 0, f"bench_holo (hilbert) failed: {err2}\n{out2}"
    assert "== Primes ==" in out2
