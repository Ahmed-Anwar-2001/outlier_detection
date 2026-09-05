"""
Groq VLM client using Qwen 3.6 27B for retail verification.
"""
import base64
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any, Dict

import httpx
from PIL import Image

try:
    from .config import VlmConfig
    from .prompts import SYSTEM_PROMPT, USER_PROMPT
except (ImportError, ValueError):
    from pipelines.vlm_groq.config import VlmConfig
    from pipelines.vlm_groq.prompts import SYSTEM_PROMPT, USER_PROMPT


class GroqVlmClient:
    def __init__(self, config: VlmConfig, api_key: str = None):
        self.config = config
        self.api_key = VlmConfig.get_api_key(api_key)
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def analyze_image(self, image_path: Path) -> Dict[str, Any]:
        """Analyze a single outlet image with caching."""
        cache_key = self._get_cache_key(image_path)
        cache_file = self.config.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        base64_data = self._encode_image(image_path)
        result = self._call_api(base64_data)

        # Only cache successful extractions
        if result.get("scene_type") != "unknown":
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
            except Exception:
                pass

        return result

    def _encode_image(self, image_path: Path) -> str:
        """Resize image to max 768px and encode to base64 JPEG."""
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            # Downscale proportionally so largest edge is max_image_dim
            w, h = img.size
            if max(w, h) > self.config.max_image_dim:
                scale = self.config.max_image_dim / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self.config.jpeg_quality)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _call_api(self, base64_data: str) -> Dict[str, Any]:
        """Call Groq chat completions endpoint with Qwen 3.6 27B."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_data}"},
                        },
                    ],
                },
            ],
            "temperature": 0.1,
            "max_tokens": 4096,
        }

        last_err = None
        for attempt in range(self.config.max_retries):
            try:
                with httpx.Client(timeout=self.config.timeout_seconds) as client:
                    resp = client.post(self.config.api_url, headers=headers, json=payload)
                    if resp.status_code == 200:
                        raw = resp.json()["choices"][0]["message"]["content"]
                        return self._parse_json_response(raw)
                    elif resp.status_code == 429:
                        time.sleep(self.config.retry_delay * (attempt + 1) * 2)
                    else:
                        last_err = f"HTTP {resp.status_code}: {resp.text}"
            except Exception as e:
                last_err = str(e)
                time.sleep(self.config.retry_delay * (attempt + 1))

        return {
            "scene_type": "unknown",
            "business_category": "unknown",
            "signboard_name": None,
            "brand_sponsors": [],
            "architectural_features": [],
            "primary_colors": [],
            "scene_description": f"Extraction error: {last_err}",
        }

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Strip <think> tags and parse JSON cleanly with error recovery."""
        import ast
        import re

        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        elif "<think>" in text:
            text = text.split("<think>", 1)[0].strip()

        # Remove markdown codeblocks
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()

        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

        # Clean trailing commas: ,} or ,]
        text_cleaned = re.sub(r",\s*([\]}])", r"\1", text)

        try:
            return json.loads(text_cleaned)
        except Exception:
            # Fallback to ast.literal_eval for python-style dicts
            try:
                # Replace JSON booleans/null with Python equivalents
                py_text = text_cleaned.replace("null", "None").replace("true", "True").replace("false", "False")
                val = ast.literal_eval(py_text)
                if isinstance(val, dict):
                    return val
            except Exception:
                pass
            # Re-raise standard JSON exception if all recovery fails
            return json.loads(text)

    def _get_cache_key(self, image_path: Path) -> str:
        stat = image_path.stat()
        raw = f"{image_path.name}_{stat.st_size}_{stat.st_mtime}_{self.config.model_name}"
        return hashlib.md5(raw.encode()).hexdigest()
