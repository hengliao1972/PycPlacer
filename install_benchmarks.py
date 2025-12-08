#!/usr/bin/env python3
"""
Benchmark Data Installer for PycPlacer

Downloads and installs standard benchmark circuits:
- ISCAS85 combinational benchmarks
- ISCAS89 sequential benchmarks  
- EPFL combinational benchmarks
- ITC99 benchmarks

Usage:
    python install_benchmarks.py           # Install all benchmarks
    python install_benchmarks.py --list    # List available benchmarks
    python install_benchmarks.py iscas85   # Install specific benchmark suite
"""

import os
import sys
import subprocess
import urllib.request
import tarfile
import zipfile
import shutil
from pathlib import Path

# Benchmark data directory
BENCHMARK_DIR = Path(__file__).parent / "benchmarks_data"

# Benchmark sources
BENCHMARKS = {
    "iscas85": {
        "description": "ISCAS85 combinational benchmarks (10 circuits)",
        "type": "download",
        "urls": [
            # ISCAS85 .bench files from various sources
            ("https://raw.githubusercontent.com/cuhk-eda/benchmarks/master/iscas85/c432.bench", "iscas85/c432.bench"),
            ("https://raw.githubusercontent.com/cuhk-eda/benchmarks/master/iscas85/c499.bench", "iscas85/c499.bench"),
            ("https://raw.githubusercontent.com/cuhk-eda/benchmarks/master/iscas85/c880.bench", "iscas85/c880.bench"),
            ("https://raw.githubusercontent.com/cuhk-eda/benchmarks/master/iscas85/c1355.bench", "iscas85/c1355.bench"),
            ("https://raw.githubusercontent.com/cuhk-eda/benchmarks/master/iscas85/c1908.bench", "iscas85/c1908.bench"),
            ("https://raw.githubusercontent.com/cuhk-eda/benchmarks/master/iscas85/c2670.bench", "iscas85/c2670.bench"),
            ("https://raw.githubusercontent.com/cuhk-eda/benchmarks/master/iscas85/c3540.bench", "iscas85/c3540.bench"),
            ("https://raw.githubusercontent.com/cuhk-eda/benchmarks/master/iscas85/c5315.bench", "iscas85/c5315.bench"),
            ("https://raw.githubusercontent.com/cuhk-eda/benchmarks/master/iscas85/c6288.bench", "iscas85/c6288.bench"),
            ("https://raw.githubusercontent.com/cuhk-eda/benchmarks/master/iscas85/c7552.bench", "iscas85/c7552.bench"),
        ],
    },
    "iscas89": {
        "description": "ISCAS89 sequential benchmarks (31 circuits, includes s38417 ★)",
        "type": "git",
        "url": "https://github.com/cuhk-eda/benchmarks.git",
        "sparse_path": "iscas89",
        "target": "iscas89",
    },
    "epfl": {
        "description": "EPFL combinational benchmarks (arithmetic + random_control)",
        "type": "git",
        "url": "https://github.com/lsils/benchmarks.git",
        "target": "epfl_benchmarks",
    },
    "itc99": {
        "description": "ITC99 benchmarks (large sequential circuits)",
        "type": "git",
        "url": "https://github.com/cuhk-eda/benchmarks.git",
        "sparse_path": "itc99",
        "target": "itc99",
    },
    "mcnc": {
        "description": "MCNC benchmarks (classic combinational circuits)",
        "type": "git",
        "url": "https://github.com/cuhk-eda/benchmarks.git",
        "sparse_path": "mcnc",
        "target": "mcnc",
    },
}


def print_banner():
    """Print welcome banner."""
    print("\n" + "=" * 60)
    print("  PycPlacer Benchmark Installer")
    print("=" * 60)


def list_benchmarks():
    """List available benchmark suites."""
    print_banner()
    print("\nAvailable benchmark suites:\n")
    
    for name, info in BENCHMARKS.items():
        status = "✓ installed" if (BENCHMARK_DIR / info.get("target", name)).exists() else "  not installed"
        print(f"  {name:<12} - {info['description']}")
        print(f"               [{status}]")
    
    print("\nUsage:")
    print("  python install_benchmarks.py <suite>    # Install specific suite")
    print("  python install_benchmarks.py all        # Install all suites")
    print("")


def download_file(url: str, dest: Path):
    """Download a file from URL."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {dest.name}...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        print("✓")
        return True
    except Exception as e:
        print(f"✗ ({e})")
        return False


def clone_git_repo(url: str, target: Path, sparse_path: str = None):
    """Clone a git repository (optionally with sparse checkout)."""
    if target.exists():
        print(f"  {target.name} already exists, skipping...")
        return True
    
    print(f"  Cloning {url}...")
    
    try:
        if sparse_path:
            # Sparse checkout for specific subdirectory
            target.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init"], cwd=target, check=True, capture_output=True)
            subprocess.run(["git", "remote", "add", "origin", url], cwd=target, check=True, capture_output=True)
            subprocess.run(["git", "config", "core.sparseCheckout", "true"], cwd=target, check=True, capture_output=True)
            
            sparse_file = target / ".git" / "info" / "sparse-checkout"
            sparse_file.parent.mkdir(parents=True, exist_ok=True)
            sparse_file.write_text(f"{sparse_path}/*\n")
            
            subprocess.run(["git", "pull", "--depth=1", "origin", "master"], cwd=target, check=True, capture_output=True)
            
            # Move files from subdirectory to target
            subdir = target / sparse_path
            if subdir.exists():
                for item in subdir.iterdir():
                    shutil.move(str(item), str(target / item.name))
                subdir.rmdir()
        else:
            # Full clone
            subprocess.run(
                ["git", "clone", "--depth=1", url, str(target)],
                check=True,
                capture_output=True
            )
        
        print(f"  ✓ Cloned to {target.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to clone: {e}")
        # Cleanup on failure
        if target.exists():
            shutil.rmtree(target)
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def install_benchmark(name: str) -> bool:
    """Install a specific benchmark suite."""
    if name not in BENCHMARKS:
        print(f"Unknown benchmark: {name}")
        print(f"Available: {', '.join(BENCHMARKS.keys())}")
        return False
    
    info = BENCHMARKS[name]
    print(f"\nInstalling {name}: {info['description']}")
    
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    
    if info["type"] == "download":
        success = True
        for url, dest in info["urls"]:
            dest_path = BENCHMARK_DIR / dest
            if dest_path.exists():
                print(f"  {dest_path.name} already exists, skipping...")
                continue
            if not download_file(url, dest_path):
                success = False
        return success
    
    elif info["type"] == "git":
        target = BENCHMARK_DIR / info["target"]
        return clone_git_repo(
            info["url"],
            target,
            info.get("sparse_path")
        )
    
    return False


def install_all():
    """Install all benchmark suites."""
    print_banner()
    print("\nInstalling all benchmark suites...\n")
    
    success = True
    for name in BENCHMARKS:
        if not install_benchmark(name):
            success = False
    
    return success


def count_files():
    """Count installed benchmark files."""
    if not BENCHMARK_DIR.exists():
        return 0, 0
    
    total_files = 0
    total_size = 0
    
    for f in BENCHMARK_DIR.rglob("*"):
        if f.is_file() and not f.name.startswith("."):
            total_files += 1
            total_size += f.stat().st_size
    
    return total_files, total_size


def main():
    if len(sys.argv) < 2:
        list_benchmarks()
        return
    
    arg = sys.argv[1].lower()
    
    if arg == "--list" or arg == "-l":
        list_benchmarks()
    elif arg == "all":
        success = install_all()
        files, size = count_files()
        print(f"\n{'='*60}")
        print(f"  Installation {'complete' if success else 'finished with errors'}")
        print(f"  Total: {files:,} files, {size/1024/1024:.1f} MB")
        print(f"{'='*60}\n")
    elif arg in BENCHMARKS:
        print_banner()
        success = install_benchmark(arg)
        if success:
            print(f"\n✓ {arg} installed successfully")
        else:
            print(f"\n✗ {arg} installation failed")
    else:
        print(f"Unknown option: {arg}")
        print(f"Available benchmarks: {', '.join(BENCHMARKS.keys())}")
        print("Use 'all' to install everything")


if __name__ == "__main__":
    main()
