"""
Single-image inference with MedGemma (no fine-tuning, no dataset).
Give it the path to one medical image (e.g. chest X-ray) and it returns a
generated description/report.

Requires (pip install):
  torch transformers>=4.50.0 pillow accelerate huggingface_hub

MedGemma is gated -- before running this, log into Hugging Face, visit
https://huggingface.co/google/medgemma-4b-it, accept the usage terms, then
either run `huggingface-cli login` in a terminal or uncomment the login()
call below.
"""

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from huggingface_hub import login
# login()

MODEL_ID = "google/medgemma-4b-it"

# System prompt shapes the style of the output. Tweak as you like.
SYSTEM_PROMPT = "You are an expert radiologist."


def load_model():
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

    model.eval()  # inference mode only -- no gradients, no training
    print("Model device:", next(model.parameters()).device)
    print("Model dtype:", next(model.parameters()).dtype)
    return model, processor


def generate_report(model, processor, frontal_image_path, lateral_image_path=None, indication="", max_new_tokens=300):
    """
    frontal_image_path: path to the frontal medical image (jpg/png). Required.
    lateral_image_path: path to the lateral image (jpg/png). Optional --
                         leave as None if you only have a frontal view.
    indication: optional short clinical context / instruction, e.g.
                "Describe this chest X-ray, noting any acute findings."
                If left as "", a sensible default prompt is used.
    """
    device = next(model.parameters()).device
    frontal_image = Image.open(frontal_image_path).convert("RGB")
    lateral_image = (
        Image.open(lateral_image_path).convert("RGB")
        if lateral_image_path is not None
        else None
    )

    user_text = indication.strip() or "Describe this medical image in detail."

    if lateral_image is not None:
        # MedGemma has no dedicated "lateral" slot like MAIRA-2 -- both views
        # are passed as separate image entries in the same message, with the
        # text explicitly labelling which image is which so the model doesn't
        # have to guess.
        user_content = [
            {
                "type": "text",
                "text": user_text
                + " The first image is the frontal view, the second image is the lateral view.",
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
    return decoded_text.strip()