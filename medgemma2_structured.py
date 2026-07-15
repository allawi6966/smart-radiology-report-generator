"""
Structured report generation with MedGemma (no fine-tuning, no dataset).
Give it the path to one medical image (e.g. chest X-ray) and it returns a
structured report with individual findings + impression, ready for verification.

Requires (pip install):
  torch transformers>=4.50.0 pillow accelerate huggingface_hub

MedGemma is gated -- before running this, log into Hugging Face, visit
https://huggingface.co/google/medgemma-4b-it, accept the usage terms, then
either run `huggingface-cli login` in a terminal or uncomment the login()
call below.
"""

import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import login
# login()

MODEL_ID = "google/medgemma-4b-it"

# Structured output schema
STRUCTURED_REPORT_SCHEMA = {
    "findings": [
        {
            "id": "finding_1",
            "finding_name": "Consolidation",
            "location": "Right lower lobe",
            "severity": "mild|moderate|severe",
            "description": "Brief clinical description",
            "confidence": 0.0  # 0-1 confidence score
        }
    ],
    "impression": "Summary of key findings and clinical significance",
    "abnormal": True  # Whether any abnormality detected
}

SYSTEM_PROMPT = """You are an expert radiologist analyzing chest X-rays.
Generate a structured report with the following JSON schema:
{
    "findings": [
        {
            "id": "finding_1",
            "finding_name": "Name of the finding (e.g., Consolidation, Pneumothorax, Cardiomegaly)",
            "location": "Anatomical location (e.g., Right lower lobe, Left hemidiaphragm)",
            "severity": "mild|moderate|severe|not_applicable",
            "description": "Clinical description of the finding",
            "confidence": 0.0 to 1.0
        }
    ],
    "impression": "Brief clinical summary and key findings",
    "abnormal": true or false
}

Requirements:
1. List ONLY findings that are clearly visible in the image
2. For each finding, provide location, severity, and confidence score
3. Do NOT hallucinate findings not supported by the image
4. If the image is normal, set abnormal to false and provide minimal findings
5. Return ONLY valid JSON, no additional text
"""


def load_model():
    """Load MedGemma model and processor."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)

    model.eval()  # inference mode only
    print("Model device:", next(model.parameters()).device)
    print("Model dtype:", next(model.parameters()).dtype)
    return model, processor


def generate_structured_report(model, processor, frontal_image_path, 
                                lateral_image_path=None, indication="", 
                                max_new_tokens=500):
    """
    Generate a structured report with individual findings.
    
    Args:
        model: Loaded MedGemma model
        processor: Loaded processor
        frontal_image_path: Path to frontal view image (jpg/png). Required.
        lateral_image_path: Path to lateral view image (jpg/png). Optional.
        indication: Optional clinical context or instruction.
        max_new_tokens: Maximum tokens to generate.
    
    Returns:
        dict: Structured report with findings and impression
    """
    device = next(model.parameters()).device
    frontal_image = Image.open(frontal_image_path).convert("RGB")
    lateral_image = (
        Image.open(lateral_image_path).convert("RGB")
        if lateral_image_path is not None
        else None
    )

    # Build user prompt
    user_text = (
        indication.strip() 
        or "Generate a structured radiology report for this chest X-ray."
    )
    
    if "JSON" not in user_text and "structured" not in user_text.lower():
        user_text += " Return as JSON only."

    # Prepare content based on available images
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
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device, dtype=model.dtype)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )
        output = output[0][input_len:]

    decoded_text = processor.decode(output, skip_special_tokens=True)
    
    # Parse JSON response
    return parse_structured_report(decoded_text)


def _strip_code_fences(text):
    """Remove ```json ... ``` or ``` ... ``` wrapper if present."""
    import re
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _sanitize_json_like(text):
    r"""
    Fix common MedGemma JSON quirks:
      - escaped underscores in keys: "finding\_name" -> "finding_name"
      - single-quoted keys/values (e.g. 'abnormal': false) -> double-quoted
      - Python-style True/False/None -> true/false/null
    This is best-effort cleanup, not a full JSON5 parser.
    """
    import re

    # Undo backslash-escaped underscores anywhere (model artifact, not valid JSON escape)
    text = text.replace("\\_", "_")

    # Convert single-quoted keys/strings to double-quoted, only outside of
    # already double-quoted strings. Simple heuristic: replace 'word': and
    # 'word' patterns since content is otherwise double-quoted.
    text = re.sub(r"'([^']*)'\s*:", r'"\1":', text)   # 'key':
    text = re.sub(r":\s*'([^']*)'", r': "\1"', text)  # : 'value'

    # Python-style literals
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)

    return text


def _normalize_finding_keys(finding):
    """Map inconsistent/varied key names (casing, spaces) to the canonical schema."""
    if not isinstance(finding, dict):
        return finding

    alias_map = {
        "id": "id",
        "finding_name": "finding_name", "finding name": "finding_name",
        "location": "location",
        "severity": "severity",
        "description": "description",
        "confidence": "confidence",
    }

    normalized = {}
    for key, value in finding.items():
        norm_key = key.strip().lower().replace("_", " ")
        # collapse to canonical via alias_map (built with spaces or underscores)
        canonical = alias_map.get(norm_key.replace(" ", "_"), alias_map.get(norm_key, key))
        normalized[canonical] = value
    return normalized


def _merge_report_pieces(parsed):
    """
    MedGemma sometimes returns a JSON *list* of separate objects instead of
    one merged object, e.g. [{"findings": [...]}, {"impression": ..., "abnormal": ...}].
    Merge any such pieces into a single report dict.
    """
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

    # Unexpected type (string, number, etc.)
    return {"impression": str(parsed), "findings": [], "abnormal": None}


def parse_structured_report(response_text):
    """
    Parse the model's response into a structured report dict.
    Handles cases where the response contains extra text before/after JSON,
    markdown code fences, a top-level JSON list instead of a dict, single
    quotes, escaped underscores, and inconsistent key casing.

    Args:
        response_text: Raw text from the model

    Returns:
        dict: Parsed structured report, or error dict if parsing fails
    """
    import re

    original_text = response_text
    response_text = _strip_code_fences(response_text.strip())

    candidates = [response_text]

    # If there's extra prose around the JSON, also try extracting the
    # outermost [...] or {...} block.
    list_match = re.search(r"\[.*\]", response_text, re.DOTALL)
    if list_match:
        candidates.append(list_match.group())
    obj_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if obj_match:
        candidates.append(obj_match.group())

    for candidate in candidates:
        for text in (candidate, _sanitize_json_like(candidate)):
            try:
                parsed = json.loads(text)
                report = _merge_report_pieces(parsed)
                return validate_report_structure(report)
            except json.JSONDecodeError:
                continue

    # If all parsing fails, return error structure
    print(f"Warning: Could not parse JSON from response:\n{original_text}")
    return {
        "error": "Failed to parse structured report",
        "raw_response": original_text,
        "findings": [],
        "impression": original_text,
        "abnormal": None
    }


def validate_report_structure(report):
    """
    Ensure the report has the required structure.
    Fill in missing fields with defaults.
    
    Args:
        report: Dict from JSON parsing
    
    Returns:
        dict: Validated report structure
    """
    if not isinstance(report, dict):
        return {
            "error": "Report is not a dictionary",
            "findings": [],
            "impression": str(report),
            "abnormal": None
        }
    
    # Ensure required fields exist
    if "findings" not in report or not isinstance(report["findings"], list):
        report["findings"] = []
    
    if "impression" not in report:
        report["impression"] = ""
    
    if "abnormal" not in report:
        report["abnormal"] = len(report["findings"]) > 0
    
    # Validate each finding has required fields
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
    
    return report


def print_structured_report(report):
    """Pretty-print the structured report."""
    if "error" in report:
        print(f"❌ Error: {report['error']}")
        print(f"Raw response: {report.get('raw_response', 'N/A')}")
        return
    
    print("\n" + "="*60)
    print("STRUCTURED RADIOLOGY REPORT")
    print("="*60)
    
    print(f"\n📋 Status: {'ABNORMAL' if report.get('abnormal') else 'NORMAL'}")
    
    findings = report.get("findings", [])
    if findings:
        print(f"\n🔍 FINDINGS ({len(findings)}):")
        print("-" * 60)
        for finding in findings:
            print(f"\n  Finding ID: {finding.get('id', 'N/A')}")
            print(f"  Type:       {finding.get('finding_name', 'N/A')}")
            print(f"  Location:   {finding.get('location', 'N/A')}")
            print(f"  Severity:   {finding.get('severity', 'N/A')}")
            print(f"  Confidence: {finding.get('confidence', 'N/A'):.2f}")
            print(f"  Description: {finding.get('description', 'N/A')}")
    else:
        print("\n✅ No significant findings detected.")
    
    impression = report.get("impression", "")
    if impression:
        print(f"\n💭 IMPRESSION:")
        print("-" * 60)
        print(f"{impression}")
    
    print("\n" + "="*60)


def save_report_json(report, output_path):
    """Save structured report to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✅ Report saved to {output_path}")


# ============================================================================
# Example usage
# ============================================================================
if __name__ == "__main__":
    
    
    frontal_path = "2.png"
    lateral_path = "1.png"
    output_path = "output/reports.json"
    
    print("Loading model...")
    model, processor = load_model()
    
    print(f"Generating structured report for: {frontal_path}")
    report = generate_structured_report(
        model, processor, 
        frontal_path, 
        lateral_image_path=lateral_path
    )
    
    print_structured_report(report)
    save_report_json(report, output_path)