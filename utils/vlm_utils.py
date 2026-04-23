from __future__ import annotations

from functools import lru_cache

import torch
from PIL import Image


DEFAULT_VLM_MODEL = "apple/FastVLM-0.5B"
IMAGE_TOKEN_INDEX = -200
MODEL_OPTIONS = {
    "FastVLM 0.5B (Quick)": {
        "id": "apple/FastVLM-0.5B",
        "backend": "fastvlm",
        "description": "Fast lightweight model for quick page summaries.",
    },
    "Qwen2.5-VL 3B (Stronger)": {
        "id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "backend": "pipeline",
        "description": "Stronger multimodal model for richer explanation and Q&A.",
    },
}


def get_model_options() -> dict[str, dict[str, str]]:
    return MODEL_OPTIONS


def resolve_model_choice(choice_label: str | None) -> tuple[str, str]:
    if choice_label and choice_label in MODEL_OPTIONS:
        selected = MODEL_OPTIONS[choice_label]
        return selected["id"], selected["backend"]

    default_config = next(iter(MODEL_OPTIONS.values()))
    return default_config["id"], default_config["backend"]


@lru_cache(maxsize=2)
def load_fastvlm(model_id: str = DEFAULT_VLM_MODEL):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    return tokenizer, model


def _prepare_fastvlm_inputs(tokenizer, model, image: Image.Image, prompt: str):
    messages = [{"role": "user", "content": f"<image>\n{prompt}"}]
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    preamble, suffix = rendered.split("<image>", 1)

    pre_ids = tokenizer(preamble, return_tensors="pt", add_special_tokens=False).input_ids
    post_ids = tokenizer(suffix, return_tensors="pt", add_special_tokens=False).input_ids
    image_token = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
    input_ids = torch.cat([pre_ids, image_token, post_ids], dim=1).to(model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)

    pixel_values = model.get_vision_tower().image_processor(
        images=image.convert("RGB"),
        return_tensors="pt",
    )["pixel_values"]
    pixel_values = pixel_values.to(model.device, dtype=model.dtype)
    return input_ids, attention_mask, pixel_values


def _generate_fastvlm_text(
    image: Image.Image,
    prompt: str,
    model_id: str,
    max_new_tokens: int,
) -> str:
    tokenizer, model = load_fastvlm(model_id)
    input_ids, attention_mask, pixel_values = _prepare_fastvlm_inputs(tokenizer, model, image, prompt)
    with torch.inference_mode():
        generated_ids = model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            images=pixel_values,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    return tokenizer.decode(generated_ids[0][input_ids.shape[1] :], skip_special_tokens=True).strip()


@lru_cache(maxsize=2)
def load_generic_vlm_pipeline(model_id: str):
    from transformers import pipeline

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return pipeline(
        "image-text-to-text",
        model=model_id,
        device_map="auto",
        torch_dtype=torch_dtype,
    )


def _generate_pipeline_text(
    image: Image.Image,
    prompt: str,
    model_id: str,
    max_new_tokens: int,
) -> str:
    pipe = load_generic_vlm_pipeline(model_id)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image.convert("RGB")},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    outputs = pipe(text=messages, max_new_tokens=max_new_tokens, return_full_text=False)
    if not outputs:
        return ""
    generated = outputs[0].get("generated_text", "")
    if isinstance(generated, list):
        text_parts = []
        for item in generated:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        return " ".join(text_parts).strip()
    return str(generated).strip()


def generate_multimodal_text(
    image: Image.Image,
    prompt: str,
    model_id: str = DEFAULT_VLM_MODEL,
    backend: str = "fastvlm",
    max_new_tokens: int = 220,
) -> str:
    if backend == "fastvlm":
        return _generate_fastvlm_text(image=image, prompt=prompt, model_id=model_id, max_new_tokens=max_new_tokens)
    return _generate_pipeline_text(image=image, prompt=prompt, model_id=model_id, max_new_tokens=max_new_tokens)


def safe_generate_multimodal_text(
    image: Image.Image,
    prompt: str,
    model_id: str = DEFAULT_VLM_MODEL,
    backend: str = "fastvlm",
    max_new_tokens: int = 220,
) -> tuple[str | None, str | None]:
    try:
        return (
            generate_multimodal_text(
                image=image,
                prompt=prompt,
                model_id=model_id,
                backend=backend,
                max_new_tokens=max_new_tokens,
            ),
            None,
        )
    except Exception as exc:
        return None, str(exc)


def explain_dashboard_image(
    image: Image.Image,
    page_hint: str | None = None,
    model_id: str = DEFAULT_VLM_MODEL,
    backend: str = "fastvlm",
    max_new_tokens: int = 220,
) -> str:
    prompt = (
        "You are looking at a screenshot of a business dashboard. "
        f"The page hint is: {page_hint or 'unknown page'}. "
        "Write a concise explanation with these section headers exactly: "
        "Theme, KPIs, Charts, Trends, Anomalies, Actions, Summary. "
        "Use short plain-English sentences. If some labels are too small to read, say so instead of guessing."
    )
    return generate_multimodal_text(
        image=image,
        prompt=prompt,
        model_id=model_id,
        backend=backend,
        max_new_tokens=max_new_tokens,
    )


def safe_explain_dashboard_image(
    image: Image.Image,
    page_hint: str | None = None,
    model_id: str = DEFAULT_VLM_MODEL,
    backend: str = "fastvlm",
) -> tuple[str | None, str | None]:
    try:
        return explain_dashboard_image(
            image=image,
            page_hint=page_hint,
            model_id=model_id,
            backend=backend,
        ), None
    except Exception as exc:
        return None, str(exc)

