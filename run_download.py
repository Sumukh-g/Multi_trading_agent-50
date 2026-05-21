"""Run download with proper output."""
import sys
import subprocess

if __name__ == "__main__":
    python_exe = r"C:\Users\cenas\AppData\Local\Programs\Python\Python311\python.exe"
    result = subprocess.run([python_exe, "-u", "download_data.py"], 
                          stdout=sys.stdout, 
                          stderr=sys.stderr)
    sys.exit(result.returncode)

