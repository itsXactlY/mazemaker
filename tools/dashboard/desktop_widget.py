#!/usr/bin/env python3
"""
Neural Memory Desktop Widget — lightweight alternative to Chrome desktop layer.
Uses pywebview (WebKitGTK) — ~50MB vs Chrome's ~500MB.

Usage:
    python desktop_widget.py                  # normal window
    python desktop_widget.py --desktop-layer  # transparent desktop background
    python desktop_widget.py --fullscreen     # fullscreen (kiosk)
"""
import argparse
import os
import sys
import time
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="https://localhost:8443/")
    parser.add_argument("--desktop-layer", action="store_true")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--opacity", type=float, default=0.4)
    args = parser.parse_args()

    # Ensure server is running
    try:
        import urllib.request
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        urllib.request.urlopen(args.url, context=ctx, timeout=3)
    except:
        print("Starting live server...")
        subprocess.Popen(
            [sys.executable, "live_server.py", "--port", "8443", "--desktop-layer"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(3)

    import webview

    window = webview.create_window(
        'Neural Memory',
        url=args.url,
        width=args.width,
        height=args.height,
        x=0, y=0,
        fullscreen=args.fullscreen,
        frameless=args.desktop_layer,
        on_top=False,
        transparent=args.desktop_layer,
        easy_drag=False,
    )

    def on_loaded():
        """Set X11 properties after window loads."""
        time.sleep(2)
        try:
            # Find our window by title
            result = subprocess.run(
                ["xdotool", "search", "--name", "Neural Memory"],
                capture_output=True, text=True
            )
            win_ids = result.stdout.strip().split("\n")
            for wid in win_ids:
                if not wid:
                    continue
                wid = wid.strip()
                if args.desktop_layer:
                    # Push to desktop layer
                    subprocess.run([
                        "xprop", "-id", wid,
                        "-f", "_NET_WM_WINDOW_TYPE", "32a",
                        "-set", "_NET_WM_WINDOW_TYPE", "_NET_WM_WINDOW_TYPE_DESKTOP"
                    ], capture_output=True)
                    subprocess.run(["xdotool", "windowlower", wid], capture_output=True)
                    print(f"Window {wid} set to DESKTOP layer")
        except Exception as e:
            print(f"X11 setup error: {e}")

    import threading
    threading.Thread(target=on_loaded, daemon=True).start()

    webview.start(debug=False)


if __name__ == "__main__":
    main()
