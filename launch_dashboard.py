"""Launch the dashboard - cross-platform launcher."""
import subprocess
import sys
import webbrowser
import time
import os

def main():
    print("="*60)
    print("LAUNCHING QUANTITATIVE TRADING DASHBOARD")
    print("="*60)
    print()
    print("Starting Streamlit server...")
    print("The dashboard will open in your browser automatically.")
    print()
    print("Press Ctrl+C to stop the server.")
    print()
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(3)
        webbrowser.open("http://localhost:8501")
    
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.start()
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "dashboard.py",
        "--server.port", "8501",
        "--browser.gatherUsageStats", "false"
    ])

if __name__ == "__main__":
    main()

