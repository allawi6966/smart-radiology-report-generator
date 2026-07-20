"""
Grounded structured report generation with MedGemma (no fine-tuning, no dataset).
Give it the path to one medical image (e.g. chest X-ray) and it returns a
structured report with individual findings + bounding boxes + impression,
then draws the boxes on the image so you can see WHERE each finding is.

Requires (pip install):
  torch transformers>=4.50.0 pillow accelerate huggingface_hub bitsandbytes>=0.46.1

MedGemma is gated -- before running this, log into Hugging Face, visit
https://huggingface.co/google/medgemma-4b-it, accept the usage terms, then
either run `huggingface-cli login` in a terminal or uncomment the login()
call below.
"""

import os
# Must be set BEFORE torch is imported so the CUDA allocator picks it up
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
import re
import torch
try:
    from json_repair import repair_json
except ImportError:
    repair_json = None  # falls back to regex-only parsing if not installed
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from huggingface_hub import login
# login()

MODEL_ID = "google/medgemma-4b-it"

# ============================================================================
# Prompt — requires bounding_box on every finding, with a worked example
# so the model has a concrete pattern to copy instead of just a spec.
# ============================================================================
SYSTEM_PROMPT = """You are an expert radiologist analyzing chest X-rays.
Generate a structured report with the following JSON schema.
bounding_box is REQUIRED on every finding — omitting it is an invalid response.

{
    "findings": [
        {
            "id": "finding_1",
            "finding_name": "Name of the finding (e.g., Consolidation, Pneumothorax, Cardiomegaly)",
            "location": "Anatomical location (e.g., Right lower lobe, Left hemidiaphragm)",
            "bounding_box": {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0},
            "severity": "mild|moderate|severe|not_applicable",
            "description": "Clinical description of the finding",
            "confidence": 0.0 to 1.0
        }
    ],
    "impression": "Brief clinical summary and key findings",
    "abnormal": true or false
}

Example of a correctly formatted finding with its bounding box:
{
    "id": "finding_1",
    "finding_name": "Cardiomegaly",
    "location": "Heart",
    "bounding_box": {"x": 0.32, "y": 0.45, "width": 0.36, "height": 0.28},
    "severity": "mild",
    "description": "The cardiac silhouette appears mildly enlarged.",
    "confidence": 0.9
}

Bounding box rules:
- Normalized 0-1 coordinates relative to the image
- x, y = top-left corner; width, height = box size
- Every finding, including normal/negative findings you choose to report, must include a bounding_box
  covering the relevant anatomical region

Requirements:
1. List ONLY findings that are clearly visible in the image
2. For each finding, provide location, bounding_box, severity, and confidence score
3. Do NOT hallucinate findings not supported by the image
4. If the image is normal, set abnormal to false and provide minimal findings
5. Do not think out loud. Do not explain your reasoning. Output JSON immediately.
6. Return ONLY valid JSON, no additional text, no markdown fences
7. Use the EXACT field names shown above: "finding_name" (not finding_type), "bounding_box" (not bounding_bbox), and "x"/"y"/"width"/"height" (not xmin/ymin/xmax/ymax)
8. Every numeric value must be a complete number (e.g. 0.0, not a bare "0.")
9. Each finding's JSON object must be complete — every field present exactly once, in the order shown above — before you move on to the next finding.
"""


def load_model():
    """Load MedGemma model in 4-bit to fit comfortably on a ~14-16GB GPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            dtype=torch.float32,
        ).to(device)

    model.eval()
    print("Model device:", next(model.parameters()).device)
    print("Model dtype:", next(model.parameters()).dtype)
    return model, processor


def generate_structured_report(model, processor, frontal_image_path,
                                lateral_image_path=None, indication="",
                                max_new_tokens=1024, verbose=True):
    """
    Generate a structured, grounded report with individual findings + boxes.

    Args:
        model: Loaded MedGemma model
        processor: Loaded processor
        frontal_image_path: Path to frontal view image (jpg/png). Required.
        lateral_image_path: Path to lateral view image (jpg/png). Optional.
        indication: Optional clinical context or instruction.
        max_new_tokens: Maximum tokens to generate (needs headroom for JSON + boxes).
        verbose: If True, print the raw model output before parsing (for debugging).

    Returns:
        dict: Structured report with findings (incl. bounding_box) and impression
    """
    device = next(model.parameters()).device
    frontal_image = Image.open(frontal_image_path).convert("RGB")
    frontal_image.thumbnail((896, 896))  # cap size to limit vision-token memory

    lateral_image = None
    if lateral_image_path is not None:
        lateral_image = Image.open(lateral_image_path).convert("RGB")
        lateral_image.thumbnail((896, 896))

    user_text = (
        indication.strip()
        or "Generate a structured, grounded radiology report with bounding boxes for this chest X-ray."
    )
    if "JSON" not in user_text and "structured" not in user_text.lower():
        user_text += " Return as JSON only."

    if lateral_image is not None:
        user_content = [
            {
                "type": "text",
                "text": user_text
                + " The first image is the frontal view, the second is the lateral view.",
            },
            {"type": "image", "image": frontal_image},
            {"type": "image", "image": lateral_image},
        ]
    else:
        user_content = [
            {"type": "text", "text": user_text},
            {"type": "image", "image": frontal_image},
        ]

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device, dtype=model.dtype)

    input_len = inputs["input_ids"].shape[-1]

    torch.cuda.empty_cache()
    with torch.inference_mode():
        # NOTE: repetition_penalty / no_repeat_ngram_size were removed here.
        # They actively break JSON generation: schemas legitimately repeat
        # the same field names ("severity", "confidence", "bounding_box", ...)
        # once per finding, and n-gram/repetition penalties punish the model
        # for reusing those tokens, forcing it to mangle or drop fields on
        # the second+ finding. Greedy decoding (do_sample=False) is already
        # deterministic and appropriate for structured extraction.
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
        output = output[0][input_len:]

    decoded_text = processor.decode(output, skip_special_tokens=True)

    if verbose:
        print("\n" + "-" * 60)
        print("RAW MODEL OUTPUT (pre-parse)")
        print("-" * 60)
        print(decoded_text)
        print("-" * 60 + "\n")

    return parse_structured_report(decoded_text)


# ============================================================================
# Robust parsing helpers
# ============================================================================
def _strip_code_fences(text):
    """Strip a ```json ... ``` or ``` ... ``` fence wrapping the text.

    Uses re.search (not re.match) with an anchored-ish pattern so it still
    works if there's incidental whitespace/newlines around the fence, and
    falls back to just stripping a bare leading/trailing ``` if the full
    fenced-block pattern doesn't match (e.g. truncated generation with no
    closing fence).
    """
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: strip a leading fence even if there's no matching closing
    # fence (or the closing fence has trailing junk after it), and strip a
    # trailing fence even if there's no leading one.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _sanitize_json_like(text):
    text = text.replace("\\_", "_")
    text = re.sub(r"'([^']*)'\s*:", r'"\1":', text)
    text = re.sub(r":\s*'([^']*)'", r': "\1"', text)
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)
    return text


def _normalize_bounding_box(box):
    """Convert xmin/ymin/xmax/ymax style boxes to x/y/width/height."""
    if not isinstance(box, dict):
        return box
    keys = {k.strip().lower(): v for k, v in box.items()}
    if all(k in keys for k in ("xmin", "ymin", "xmax", "ymax")):
        try:
            xmin, ymin, xmax, ymax = (float(keys["xmin"]), float(keys["ymin"]),
                                       float(keys["xmax"]), float(keys["ymax"]))
            return {"x": xmin, "y": ymin, "width": xmax - xmin, "height": ymax - ymin}
        except (TypeError, ValueError):
            return box
    return box


def _normalize_finding_keys(finding):
    if not isinstance(finding, dict):
        return finding
    alias_map = {
        "id": "id",
        "finding_name": "finding_name", "finding name": "finding_name",
        "finding_type": "finding_name", "finding type": "finding_name",
        "location": "location",
        "bounding_box": "bounding_box", "bounding box": "bounding_box",
        "bounding_bbox": "bounding_box", "bounding bbox": "bounding_box", "bbox": "bounding_box",
        "severity": "severity",
        "description": "description",
        "confidence": "confidence",
    }
    normalized = {}
    for key, value in finding.items():
        norm_key = key.strip().lower().replace("_", " ")
        canonical = alias_map.get(norm_key.replace(" ", "_"), alias_map.get(norm_key, key))
        if canonical == "bounding_box":
            value = _normalize_bounding_box(value)
        normalized[canonical] = value
    return normalized


def _merge_report_pieces(parsed):
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        merged = {"findings": [], "impression": "", "abnormal": None}
        for piece in parsed:
            if not isinstance(piece, dict):
                continue
            if "findings" in piece and isinstance(piece["findings"], list):
                merged["findings"].extend(piece["findings"])
            if "impression" in piece:
                merged["impression"] = piece["impression"]
            if "abnormal" in piece:
                merged["abnormal"] = piece["abnormal"]
        return merged
    return {"impression": str(parsed), "findings": [], "abnormal": None}


def parse_structured_report(response_text):
    """Parse the model's response into a structured report dict, tolerating
    leading chain-of-thought text, code fences, JSON lists, single quotes, etc."""
    original_text = response_text

    # Strip any leading chain-of-thought / preamble before the JSON begins
    json_start = response_text.find('{')
    list_start = response_text.find('[')
    starts = [s for s in (json_start, list_start) if s != -1]
    if starts:
        response_text = response_text[min(starts):]

    response_text = _strip_code_fences(response_text.strip())

    # IMPORTANT: try the full OBJECT match before the bare-ARRAY match.
    # A response like `{"findings": [...], "impression": ..., "abnormal": ...}`
    # will always contain an inner "[...]" (the findings array) that also
    # matches the array regex and parses "successfully" on its own -- but
    # merging a bare list of finding-dicts loses "impression"/"abnormal"
    # entirely and silently produces an empty-looking report. Trying the
    # object match first means we grab the real, complete report whenever
    # one is present, and only fall back to the bare array if there's no
    # enclosing object at all.
    candidates = [response_text]
    obj_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if obj_match:
        candidates.append(obj_match.group())
    list_match = re.search(r"\[.*\]", response_text, re.DOTALL)
    if list_match:
        candidates.append(list_match.group())

    for candidate in candidates:
        for text in (candidate, _sanitize_json_like(candidate)):
            try:
                parsed = json.loads(text)
                report = _merge_report_pieces(parsed)
                # Guard: a bare list of finding-dicts parses without error,
                # but _merge_report_pieces only pulls out keys literally
                # named "findings"/"impression"/"abnormal" from each item --
                # a list of finding objects (which don't have those keys)
                # merges into an empty shell. Don't accept that as success;
                # keep trying remaining candidates (e.g. the full object).
                if isinstance(parsed, list) and not report.get("findings") and not report.get("impression"):
                    continue
                return validate_report_structure(report)
            except json.JSONDecodeError:
                continue

    # Last resort: dedicated JSON repair (handles missing commas/quotes,
    # truncated numbers like "0.", trailing commas, etc. far more robustly
    # than hand-written regex).
    if repair_json is not None:
        for candidate in candidates:
            try:
                fixed = repair_json(candidate)
                parsed = json.loads(fixed)
                report = _merge_report_pieces(parsed)
                print("ℹ️  Recovered via json_repair (raw output was malformed JSON).")
                return validate_report_structure(report)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

    print(f"Warning: Could not parse JSON from response:\n{original_text}")
    return {
        "error": "Failed to parse structured report",
        "raw_response": original_text,
        "findings": [],
        "impression": original_text,
        "abnormal": None,
    }


def validate_report_structure(report):
    if not isinstance(report, dict):
        return {
            "error": "Report is not a dictionary",
            "findings": [],
            "impression": str(report),
            "abnormal": None,
        }

    if "findings" not in report or not isinstance(report["findings"], list):
        report["findings"] = []
    if "impression" not in report:
        report["impression"] = ""
    if "abnormal" not in report:
        report["abnormal"] = len(report["findings"]) > 0

    for i, finding in enumerate(report["findings"]):
        if not isinstance(finding, dict):
            continue
        report["findings"][i] = finding = _normalize_finding_keys(finding)
        if "id" not in finding:
            finding["id"] = f"finding_{i+1}"
        if "finding_name" not in finding:
            finding["finding_name"] = "Unknown"
        if "location" not in finding:
            finding["location"] = "Unspecified"
        if "severity" not in finding:
            finding["severity"] = "not_applicable"
        if "description" not in finding:
            finding["description"] = ""
        if "confidence" not in finding:
            finding["confidence"] = 0.5
        # bounding_box intentionally NOT defaulted — missing means the model
        # didn't ground this finding, which validate_grounded() below flags.

    return report


def is_degenerate_finding(finding):
    """
    True if a finding is just the placeholder/default shell left behind
    by validate_report_structure() rather than real model content — i.e.
    json_repair patched the JSON *shape* but no actual clinical content
    survived. Filtering these out stops the pipeline from silently
    reporting empty findings as if they were real.
    """
    if not isinstance(finding, dict):
        return True
    name_missing = finding.get("finding_name", "Unknown") in ("Unknown", "", None)
    desc_missing = not finding.get("description")
    box_missing = not isinstance(finding.get("bounding_box"), dict)
    return name_missing and desc_missing and box_missing


def clean_report(report):
    """Drop degenerate placeholder findings and fix up abnormal/impression
    accordingly. Call this after generation, before printing/saving."""
    findings = report.get("findings", [])
    real_findings = [f for f in findings if not is_degenerate_finding(f)]
    dropped = len(findings) - len(real_findings)

    if dropped:
        print(f"⚠️  Dropped {dropped} degenerate/placeholder finding(s) with no real content.")

    report["findings"] = real_findings
    if not real_findings:
        report["abnormal"] = False
        if not report.get("impression") or report.get("impression") == "":
            report["impression"] = "No valid findings could be generated from the model output."

    return report


def validate_grounded(report):
    """True only if every finding has a bounding_box with all 4 keys."""
    findings = report.get("findings", [])
    if not findings:
        return False
    for f in findings:
        box = f.get("bounding_box")
        if not isinstance(box, dict):
            return False
        if not all(k in box for k in ("x", "y", "width", "height")):
            return False
    return True


# ============================================================================
# Drawing boxes on the image — this is the part that actually shows "where"
# ============================================================================
def draw_bounding_boxes(image_path, report, output_path="output/annotated.png"):
    """Draw each finding's bounding_box onto the original image and save it."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    img_w, img_h = image.size

    colors = {
        "mild": "yellow",
        "moderate": "orange",
        "severe": "red",
        "not_applicable": "deepskyblue",
    }

    drawn = 0
    for finding in report.get("findings", []):
        box = finding.get("bounding_box")
        if not isinstance(box, dict) or not all(k in box for k in ("x", "y", "width", "height")):
            continue

        x = box["x"] * img_w
        y = box["y"] * img_h
        w = box["width"] * img_w
        h = box["height"] * img_h

        color = colors.get(finding.get("severity", "mild"), "yellow")
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)

        label = f"{finding.get('finding_name', '')} ({finding.get('confidence', 0):.0%})"
        draw.text((x, max(0, y - 12)), label, fill=color)
        drawn += 1

    image.save(output_path)
    print(f"✅ Annotated image saved to {output_path} ({drawn} box(es) drawn)")
    return image


def print_structured_report(report):
    if "error" in report:
        print(f"❌ Error: {report['error']}")
        print(f"Raw response: {report.get('raw_response', 'N/A')}")
        return

    print("\n" + "=" * 60)
    print("STRUCTURED RADIOLOGY REPORT")
    print("=" * 60)
    print(f"\n📋 Status: {'ABNORMAL' if report.get('abnormal') else 'NORMAL'}")

    findings = report.get("findings", [])
    if findings:
        print(f"\n🔍 FINDINGS ({len(findings)}):")
        print("-" * 60)
        for finding in findings:
            print(f"\n  Finding ID: {finding.get('id', 'N/A')}")
            print(f"  Type:       {finding.get('finding_name', 'N/A')}")
            print(f"  Location:   {finding.get('location', 'N/A')}")
            box = finding.get("bounding_box")
            print(f"  Box:        {box if box else 'MISSING'}")
            print(f"  Severity:   {finding.get('severity', 'N/A')}")
            conf = finding.get('confidence', 'N/A')
            conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else str(conf)
            print(f"  Confidence: {conf_str}")
            print(f"  Description: {finding.get('description', 'N/A')}")
    else:
        print("\n✅ No significant findings detected.")

    impression = report.get("impression", "")
    if impression:
        print(f"\n💭 IMPRESSION:")
        print("-" * 60)
        print(f"{impression}")

    print("\n" + "=" * 60)


def save_report_json(report, output_path):
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✅ Report saved to {output_path}")


# ============================================================================
# Example usage
# ============================================================================
if __name__ == "__main__":

    frontal_path = "2.png"
    lateral_path = "1.png"
    output_dir = "output"
    report_path = os.path.join(output_dir, "reports_grounded.json")
    annotated_path = os.path.join(output_dir, "annotated.png")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading model...")
    model, processor = load_model()

    print(f"Generating grounded structured report for: {frontal_path}")
    report = generate_structured_report(
        model, processor,
        frontal_path,
        lateral_image_path=lateral_path,
    )

    if not validate_grounded(report):
        print("⚠️  Missing bounding boxes on one or more findings — retrying once.")
        report = generate_structured_report(
            model, processor, frontal_path,
            lateral_image_path=lateral_path,
            max_new_tokens=1024,
        )

    report = clean_report(report)

    print_structured_report(report)
    save_report_json(report, report_path)

    if report.get("findings"):
        draw_bounding_boxes(frontal_path, report, annotated_path)
    else:
        print("No findings to draw.")