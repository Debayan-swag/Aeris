"""
Fast Start Script for Aeris
Optimized launcher with cache warmup
"""
import subprocess
import sys
import os
from pathlib import Path

def print_banner():
    banner = """
    ╔═══════════════════════════════════════════╗
    ║          A E R I S   F A S T              ║
    ║     Optimized Satellite Intelligence      ║
    ╚═══════════════════════════════════════════╝
    """
    print(banner)

def main():
    print_banner()
    
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("🚀 Starting Aeris with optimizations...\n")
    
    # Optional: Warmup cache (comment out if you want fastest startup)
    # print("⚡ Warming up cache for faster performance...")
    # subprocess.run([sys.executable, "warmup_cache.py"], check=False)
    # print()
    
    print("Launching Streamlit...\n")
    print("-" * 50)
    
    try:
        # Launch with all optimizations
        subprocess.run([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "GUI.py",
            "--server.port=8501",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
            "--server.fileWatcherType=none",
            "--server.runOnSave=false",
            "--client.showErrorDetails=false",
            "--client.toolbarMode=minimal",
            "--runner.fastReruns=true",
            "--logger.level=error",
            "--global.developmentMode=false"
        ])
    except KeyboardInterrupt:
        print("\n\nAeris stopped")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
