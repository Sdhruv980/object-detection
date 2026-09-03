"""
bill_scanner.py — Video Bill Extractor
Reads a video where bills/receipts are shown one by one.
Detects stable "bill frames", sends each to Gemini Vision,
and outputs structured JSON with every bill's data.

Usage:
  python bill_scanner.py --video inputs/bills.mp4
  python bill_scanner.py --video inputs/bills.mp4 --output outputs/bills.json --debug
"""

import argparse
import base64
import json
import os
import sys
import time
import cv2
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
GEMINI_URL     = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)

BILL_PROMPT = """You are an expert AI document and bill OCR extraction system.
Carefully examine this image. If it contains ANY bill, invoice, retail receipt, cash memo, or courier shipping label (such as Delhivery, Amazon, Flipkart, BlueDart, restaurant, store, etc.), extract ALL details with high accuracy.

Return ONLY valid JSON (no markdown formatting, no code blocks):
{
  "is_bill": true,
  "doc_type": "Invoice / Retail Bill / Delivery Label / Cash Receipt",
  "vendor_name": "Name of Store, Seller, Courier, or Company",
  "bill_number": "Bill No, Invoice No, or AWB / Tracking No if visible",
  "date": "Date on document if shown",
  "customer_name": "Recipient / Customer name if visible",
  "customer_address": "Destination City, State, PIN, or Address if visible",
  "items": [
    {"name": "Product or Item description", "qty": "quantity", "price": "unit price", "amount": "line total"}
  ],
  "subtotal": "Subtotal amount with currency symbol or null",
  "tax": "Tax / GST / VAT amount if shown or null",
  "discount": "Discount if shown or null",
  "total": "Final Total amount with currency symbol (e.g. INR 1098, ₹549.00)",
  "payment_method": "Pre-paid / COD / Cash / Card / UPI if visible",
  "notes": "Any other relevant details or null"
}

If this image is blurry or contains NO document/bill/receipt at all, return exactly:
{"is_bill": false}"""


# ---------------------------------------------------------------------------
# Frame extraction — detect stable "bill" frames
# ---------------------------------------------------------------------------

def extract_bill_frames(video_path: str, debug: bool = False) -> list:
    """
    Read the video and extract frames where a bill is steadily held.
    Strategy:
      - Sample every N frames
      - Compute mean pixel difference with previous sample
      - When diff drops below STABLE_THRESH after being above MOTION_THRESH
        → treat it as a new stable bill frame
      - Skip frames too close to the last captured one
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total        = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_step  = max(1, int(fps / 5))   # 5 samples per second
    stable_gap   = int(fps * 2.5)         # min 2.5 s between captures
    motion_thresh = 8.0                   # mean pixel diff → "moving"
    stable_thresh = 3.5                   # mean pixel diff → "stable"

    frames     = []   # (frame_idx, frame_bgr)
    prev_gray  = None
    in_motion  = False
    last_cap   = -stable_gap
    idx        = 0

    print(f"[INFO] Video: {total} frames @ {fps:.1f} fps — sampling every {sample_step} frame(s)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % sample_step == 0:
            small = cv2.resize(frame, (320, 240))
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray  = cv2.GaussianBlur(gray, (15, 15), 0)

            if prev_gray is not None:
                diff      = cv2.absdiff(prev_gray, gray)
                mean_diff = float(diff.mean())

                if mean_diff > motion_thresh:
                    in_motion = True

                elif in_motion and mean_diff < stable_thresh:
                    # Transitioned from motion → stable
                    if (idx - last_cap) >= stable_gap:
                        frames.append((idx, frame.copy()))
                        last_cap  = idx
                        in_motion = False
                        ts = idx / fps
                        print(f"[CAPTURE] frame {idx} @ {ts:.1f}s  (diff={mean_diff:.2f})")
                        if debug:
                            dbg_path = f"outputs/debug_frame_{len(frames)}.jpg"
                            cv2.imwrite(dbg_path, frame)

            prev_gray = gray

        idx += 1

    cap.release()

    # Fallback: if no motion detected (e.g. static video), sample every 3 s
    if not frames:
        print("[WARN] No motion detected — falling back to periodic sampling (every 3s)")
        cap = cv2.VideoCapture(video_path)
        step = int(fps * 3)
        idx  = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frames.append((idx, frame.copy()))
            idx += 1
        cap.release()

    print(f"[INFO] Captured {len(frames)} candidate frame(s) for Gemini analysis")
    return frames


# ---------------------------------------------------------------------------
# Gemini Vision call
# ---------------------------------------------------------------------------

def frame_to_base64(frame) -> str:
    """Encode a BGR frame to JPEG base64."""
    # Resize to 1280px max side for API efficiency
    h, w = frame.shape[:2]
    if max(h, w) > 1280:
        scale = 1280 / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def analyze_with_gemini(frame, frame_idx: int) -> dict | None:
    """Send one frame to Gemini and return parsed bill JSON (or None)."""
    if not GEMINI_API_KEY:
        print("[ERROR] GEMINI_API_KEY not set in .env")
        return None

    b64 = frame_to_base64(frame)
    payload = {
        "contents": [{
            "parts": [
                {"text": BILL_PROMPT},
                {"inline_data": {"mime_type": "image/jpeg", "data": b64}}
            ]
        }],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1000}
    }

    try:
        resp = requests.post(
            GEMINI_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()

        # Strip markdown code fences if present
        if text.startswith("`"):
            text = text.split("`")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        result = json.loads(text)
        result["_frame_idx"] = frame_idx
        return result

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Gemini API request failed for frame {frame_idx}: {e}")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[ERROR] Failed to parse Gemini response for frame {frame_idx}: {e}")
        print(f"        Raw response: {text[:200] if 'text' in dir() else 'N/A'}")
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Extract bill data from a video using Gemini Vision")
    parser.add_argument("--video",  required=True, help="Path to input video file")
    parser.add_argument("--output", default="outputs/bills.json", help="Output JSON path")
    parser.add_argument("--debug",  action="store_true", help="Save debug frame images to outputs/")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs("outputs", exist_ok=True)

    if not os.path.isfile(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        sys.exit(1)

    # 1. Extract candidate frames
    frames = extract_bill_frames(args.video, debug=args.debug)

    if not frames:
        print("[ERROR] No frames extracted from video.")
        sys.exit(1)

    # 2. Analyze each frame with Gemini
    bills      = []
    skipped    = 0
    bill_count = 0

    for i, (fidx, frame) in enumerate(frames):
        print(f"\n[GEMINI] Analyzing frame {i+1}/{len(frames)} (video frame #{fidx})...")
        result = analyze_with_gemini(frame, fidx)

        if result is None:
            skipped += 1
            continue

        if not result.get("is_bill", False):
            print(f"  → Not a bill, skipped.")
            skipped += 1
            continue

        bill_count += 1
        result["bill_index"] = bill_count
        bills.append(result)
        print(f"  → Bill #{bill_count}: {result.get('vendor_name','?')}  Total: {result.get('total','?')}")

        # Respect Gemini free tier: 15 req/min
        if i < len(frames) - 1:
            time.sleep(1.2)

    # 3. Save JSON
    output = {
        "video": args.video,
        "total_bills": len(bills),
        "bills": bills
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Found {len(bills)} bill(s), skipped {skipped} non-bill frame(s).")
    print(f"[DONE] Results saved to: {args.output}")

    # 4. Print summary table
    if bills:
        print("\n" + "="*60)
        print(f"{'#':<4} {'Vendor':<25} {'Date':<12} {'Total':<12}")
        print("="*60)
        for b in bills:
            print(f"{b['bill_index']:<4} {str(b.get('vendor_name','?'))[:24]:<25} "
                  f"{str(b.get('date','?'))[:11]:<12} {str(b.get('total','?')):<12}")
        print("="*60)


if __name__ == "__main__":
    main()
