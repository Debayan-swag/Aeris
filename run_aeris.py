import os
import sys
import subprocess
from pathlib import Path


def print_banner():
    """Display Aeris banner."""
    banner = """
    ╔═══════════════════════════════════════════╗
    ║                                           ║
    ║              A E R I S                    ║
    ║                                           ║
    ║     Satellite Intelligence Platform       ║
    ║                                           ║
    ╚═══════════════════════════════════════════╝
    """
    print(banner)


def check_env():
    """Quick environment check."""
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key or api_key == "your_nvidia_api_key_here":
        print("Warning: NVIDIA_API_KEY not configured in .env file")
        print("Some features may not work properly.\n")
        return False
    return True


def check_files():
    """Quick file existence check."""
    project_dir = Path(__file__).parent
    
    required_files = [
        project_dir / "GUI.py",
        project_dir / "sentinel-2-processed.parquet",
        project_dir / "remoteclip_embeddings" / "remoteclip.faiss",
    ]
    
    missing = []
    for file in required_files:
        if not file.exists():
            missing.append(file.name)
    
    if missing:
        print(f"Warning: Missing files: {', '.join(missing)}")
        print("Run python setup.py for detailed diagnostics.\n")
        return False
    
    return True


def launch_streamlit():
    """Launch Streamlit application."""
    print("-" * 60 + "\n")
    print("Launching Aeris...\n")
    print("-" * 60 + "\n")
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "GUI.py",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
        ])
    except KeyboardInterrupt:
        print("\n\nAeris stopped gracefully.")
    except Exception as e:
        print(f"\nError launching Aeris: {e}")
        print("   Make sure streamlit is installed: pip install streamlit")
        sys.exit(1)


def main():
    """Main launcher function."""
    print_banner()
    
    # Quick checks
    env_ok = check_env()
    files_ok = check_files()
    
    if not env_ok or not files_ok:
        response = input("\nSome checks failed. Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("\nRun 'python setup.py' for detailed diagnostics and setup.")
            sys.exit(1)
    
    # Launch
    launch_streamlit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nLaunch interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nLaunch failed: {e}")
        sys.exit(1)
