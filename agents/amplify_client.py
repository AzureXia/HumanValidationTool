# -------- in agents/amplify_client.py --------
from typing import List, Dict, Optional, Any
import os, json, re, requests
from tenacity import retry, wait_exponential, stop_after_attempt
from dotenv import load_dotenv

load_dotenv()

class AmplifyClient:
    def __init__(self, model: Optional[str] = None):
        self.api_key = os.getenv("AMPLIFY_API_KEY")
        self.base_url = os.getenv("AMPLIFY_API_URL")
        if not self.api_key:  raise RuntimeError("Missing AMPLIFY_API_KEY")
        if not self.base_url: raise RuntimeError("Missing AMPLIFY_API_URL")
        self.model = model or os.getenv("AMPLIFY_MODEL", "gpt-4o-mini")
        # Vanderbilt uses Authorization; keep configurable just in case
        self.header_name = os.getenv("AMPLIFY_HEADER_NAME", "Authorization")

    def _add_auth(self, headers: Dict[str, str]) -> Dict[str, str]:
        if self.header_name.lower() == "authorization":
            headers[self.header_name] = f"Bearer {self.api_key}"
        else:
            headers[self.header_name] = self.api_key
        return headers

    def _unpack_any(self, obj: Any) -> Optional[str]:
        """Find assistant text in many possible shapes."""
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            # OpenAI-like
            if "choices" in obj:
                try:
                    return obj["choices"][0]["message"]["content"]
                except Exception:
                    pass
            # Common keys
            for k in ("text", "content", "output_text"):
                v = obj.get(k)
                if isinstance(v, str):
                    return v
            # Vanderbilt: {"data": {...}} or {"data":"..."}
            if "data" in obj:
                t = self._unpack_any(obj["data"])
                if t: return t
            # Some variants: {"output":[{"content":"..."}]}
            if "output" in obj and isinstance(obj["output"], list):
                for it in obj["output"]:
                    t = self._unpack_any(it)
                    if t: return t
            # last resort: search values
            for v in obj.values():
                t = self._unpack_any(v)
                if t: return t
        if isinstance(obj, list):
            for it in obj:
                t = self._unpack_any(it)
                if t: return t
        return None

    def _parse_sse(self, text: str) -> Optional[str]:
        """
        Aggregate SSE 'data: {...}' lines. Concatenate any 'content'/'delta'/'text' fields.
        """
        lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("data:")]
        if not lines:
            return None
        chunks = []
        for ln in lines:
            body = ln[len("data:"):].strip()
            if not body or body == "[DONE]":
                continue
            try:
                ev = json.loads(body)
            except Exception:
                continue
            # Try common keys inside each event
            for k in ("content", "delta", "text"):
                val = ev.get(k)
                if isinstance(val, str):
                    chunks.append(val)
                    break
        out = "".join(chunks).strip()
        return out or None

    @retry(wait=wait_exponential(min=1, max=8), stop=stop_after_attempt(3))
    def chat(self, messages: List[Dict], temperature: float = 0.2, max_tokens: int = 800) -> str:
        headers = self._add_auth({"Content-Type": "application/json"})
        # Vanderbilt /chat payload
        if self.base_url.rstrip("/").endswith("/chat"):
            payload = {
                "data": {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "messages": messages,
                    "options": {"skipRag": True, "model": {"id": self.model}},
                    "dataSources": []
                }
            }
        else:  # OpenAI-style fallback
            payload = {"model": self.model, "messages": messages,
                       "temperature": temperature, "max_tokens": max_tokens}

        r = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"Amplify error {r.status_code}: {r.text}")

        # 1) Try JSON first
        try:
            data = r.json()
            txt = self._unpack_any(data)
            if txt:
                return txt
        except Exception:
            pass

        # 2) Try SSE aggregation (text/event-stream buffers)
        txt = self._parse_sse(r.text)
        if txt:
            return txt

        # 3) Fallback: raw text (let callers regex/JSON-extract)
        return r.text
