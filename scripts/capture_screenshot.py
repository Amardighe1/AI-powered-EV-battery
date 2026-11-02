"""Capture high-resolution screenshot of the demo web application during runaway."""

import os
import sys
import time
import subprocess
import urllib.request
import shutil

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_PATH = os.path.join(REPO_ROOT, "plots", "runaway_screenshot.png")
ARTIFACT_PATH = "C:\\Users\\amard\\.gemini\\antigravity\\brain\\c89fc909-89ad-46fd-90e4-bf78aa2764c4\\runaway_screenshot.png"


def find_browser():
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise RuntimeError("No compatible headless browser (Edge/Chrome) found on system.")


def main():
    browser_exe = find_browser()
    print(f"Using browser: {browser_exe}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    # 1. Start server process
    print("Starting FastAPI app server...")
    proc = subprocess.Popen(
        [sys.executable, "-u", "app.py"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # 2. Wait for server readiness
    url = "http://127.0.0.1:8000"
    ready = False
    for attempt in range(40):
        time.sleep(0.5)
        try:
            with urllib.request.urlopen(f"{url}/scenarios", timeout=2) as resp:
                if resp.status == 200:
                    ready = True
                    print(f"Server is responding at {url} (attempt {attempt+1})")
                    break
        except Exception as e:
            pass

    if not ready:
        proc.kill()
        out, err = proc.communicate()
        print("Server stdout:", out.decode('latin1'))
        print("Server stderr:", err.decode('latin1'))
        raise RuntimeError("FastAPI server failed to start within timeout.")

    print("FastAPI server ready! Loading runaway scenario & taking screenshot...")

    # 3. Request URL: overcharge_runaway at t=580s (pre-runaway Early Warning with Red Banner)
    target_url = f"{url}/?scenario=overcharge_runaway&t=580"

    cmd = [
        browser_exe,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--hide-scrollbars",
        "--window-size=1500,1050",
        "--virtual-time-budget=6000",
        f"--screenshot={OUTPUT_PATH}",
        target_url,
    ]

    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        print("Browser returncode:", res.returncode)
    except Exception as e:
        print("Browser error:", e)
    finally:
        # 4. Stop server cleanly
        proc.kill()
        proc.wait()

    # 5. Verify output file
    if os.path.exists(OUTPUT_PATH) and os.path.getsize(OUTPUT_PATH) > 1000:
        size_kb = os.path.getsize(OUTPUT_PATH) / 1024
        print(f"SUCCESS: Screenshot saved to {OUTPUT_PATH} ({size_kb:.1f} KB)")
        try:
            shutil.copyfile(OUTPUT_PATH, ARTIFACT_PATH)
            print(f"Copied screenshot to artifact directory: {ARTIFACT_PATH}")
        except Exception as e:
            print(f"Notice: artifact copy: {e}")
    else:
        print(f"ERROR: Screenshot file not generated or empty at {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
