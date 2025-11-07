import argparse
import json
import re


TIME_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})\s+-->\s+(\d{2}):(\d{2}):(\d{2}),(\d{3})$")


def hms_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def normalize_text(lines):
    # Join multi-line subtitle text into a single sentence
    text = " ".join(line.strip() for line in lines if line.strip())
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_srt(srt_content: str):
    sentences = []
    # Split blocks by blank lines
    blocks = re.split(r"\r?\n\r?\n+", srt_content.strip(), flags=re.MULTILINE)
    for block in blocks:
        lines = [l for l in block.splitlines() if l is not None]
        if not lines:
            continue
        # Typical SRT block: index, time, text...
        # Some files may omit numeric index; handle both.
        if len(lines) >= 2 and TIME_RE.match(lines[1]):
            time_line_idx = 1
            text_lines = lines[2:]
        elif TIME_RE.match(lines[0]):
            time_line_idx = 0
            text_lines = lines[1:]
        else:
            # Unrecognized block; skip
            continue

        m = TIME_RE.match(lines[time_line_idx])
        if not m:
            continue
        start = hms_to_seconds(m.group(1), m.group(2), m.group(3), m.group(4))
        end = hms_to_seconds(m.group(5), m.group(6), m.group(7), m.group(8))
        text = normalize_text(text_lines)
        if not text:
            continue

        sentences.append({
            "sentence": text,
            "starttime": start,
            "endtime": end,
            "verbs": []
        })

    return {"sentences": sentences}


def main():
    parser = argparse.ArgumentParser(description="Convert SRT to project JSON format")
    parser.add_argument("input_srt", type=str, help="Path to .srt file")
    parser.add_argument("output_json", type=str, help="Path to output .json (e.g., output_data/VideoName.v4.json)")
    args = parser.parse_args()

    with open(args.input_srt, "r", encoding="utf-8", errors="replace") as f:
        srt = f.read()

    data = parse_srt(srt)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()


