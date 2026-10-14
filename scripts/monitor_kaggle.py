"""Real-Time Monitor for Kaggle Kernel Execution.
Polls status and logs so you always know exactly what is happening without guessing.

Usage:
    python scripts/monitor_kaggle.py
    python scripts/monitor_kaggle.py --interval 15 --download_on_finish
"""
import argparse
import datetime
import os
import subprocess
import sys
import time


def check_status(kernel_slug):
    try:
        import kaggle
        res = kaggle.api.kernels_status(kernel_slug)
        status = str(getattr(res, "status", getattr(res, "get", lambda k, d=None: d)("status", "UNKNOWN")))
        failure_msg = str(getattr(res, "failure_message", getattr(res, "get", lambda k, d=None: d)("failureMessage", "")))
        return status, failure_msg
    except Exception as e:
        return "ERROR_API", str(e)


def download_outputs(kernel_slug, out_dir="kaggle_outputs"):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[Monitor] Downloading kernel output artifacts to ./{out_dir}...", flush=True)
    try:
        subprocess.run(["kaggle", "kernels", "output", kernel_slug, "-p", out_dir], check=True)
        print(f"[Monitor] Artifacts downloaded successfully to ./{out_dir}!\n", flush=True)
    except Exception as e:
        print(f"[Monitor] Error downloading outputs: {e}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Monitor a running Kaggle kernel with live status updates")
    parser.add_argument("--kernel", default="udayraj1238/reid-attack-anatomy-runner", help="Kernel slug")
    parser.add_argument("--interval", type=int, default=15, help="Poll interval in seconds")
    parser.add_argument("--download_on_finish", action="store_true", help="Automatically download outputs when complete")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print(f"Kaggle Kernel Monitor: {args.kernel}")
    print(f"Poll Interval: {args.interval}s | Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65 + "\n")

    start_time = time.time()
    last_status = None

    try:
        while True:
            now_str = datetime.datetime.now().strftime("%H:%M:%S")
            elapsed = int(time.time() - start_time)
            mins, secs = divmod(elapsed, 60)
            elapsed_str = f"{mins}m {secs:02d}s"

            status, msg = check_status(args.kernel)

            status_color = status
            if status == "RUNNING":
                badge = "[RUNNING]"
            elif status in ("COMPLETE", "COMPLETED"):
                badge = "[COMPLETED]"
            elif "ERROR" in status or "FAIL" in status:
                badge = "[FAILED]"
            else:
                badge = f"[{status}]"

            print(f"[{now_str}] {badge:14s} | Elapsed: {elapsed_str} | Kernel: {args.kernel}", flush=True)

            if status in ("COMPLETE", "COMPLETED"):
                print(f"\n🎉 Kernel execution completed successfully after {elapsed_str}!", flush=True)
                if args.download_on_finish:
                    download_outputs(args.kernel)
                break

            if "FAIL" in status or "ERROR" in status or "CANCEL" in status:
                print(f"\n⚠️ Kernel stopped or failed with status: {status}", flush=True)
                if msg:
                    print(f"Details: {msg}", flush=True)
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n[Monitor] Monitoring stopped by user (kernel is still running in cloud if active).")


if __name__ == "__main__":
    main()
