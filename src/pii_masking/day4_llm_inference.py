"""Day 4 LLM prompting and output parsing pipeline."""

import json
import os
import re
import string
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "llm_template_C.yaml"
with open(_CONFIG_PATH, encoding="utf-8") as _f:
    _llm_config = yaml.safe_load(_f)

SYSTEM_PROMPT: str = _llm_config["system_prompt"]
_TEMPERATURE: float = float(_llm_config["temperature"])

class LLMPIIPipeline:
    """LLaMA-cpp inference pipeline for zero-shot PII span extraction.

    Prompts the model with a structured JSON request, then parses the response
    through a three-stage fallback (direct JSON parse -> brace extraction ->
    regex field recovery) before aligning named entities back to IOB2 token tags.
    """

    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int = 4, n_gpu_layers: int = 0):
        """Load a GGUF model via llama-cpp-python.

        n_gpu_layers=0 runs fully on CPU; set > 0 to offload layers to GPU.
        """
        from llama_cpp import Llama
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads,
                         n_gpu_layers=n_gpu_layers, verbose=False)
        self._parse_failure_log: list[dict] = []

    def _build_prompt(self, sentence: str) -> str:
        return (
            "<|start_header_id|>system<|end_header_id|>\n"
            + SYSTEM_PROMPT
            + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            "Identify all person names and email addresses in the following sentence.\n"
            "Return ONLY this JSON structure with no other text:\n"
            "{\"names\": [\"...\"], \"emails\": [\"...\"]}\n"
            "\n"
            "Sentence: " + sentence + "\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )

    def _parse_response(
        self, response: str, sentence: str = "", idx: int = -1
    ) -> tuple[list[str], list[str], bool]:
        """Parse LLM response text into (names, emails, parse_ok).

        Three-stage fallback: (1) direct json.loads on cleaned text,
        (2) extract largest {...} brace span and retry, (3) regex field
        recovery for truncated/malformed JSON.  Returns ([], [], False)
        if all stages fail.
        """
        text = response.strip()

        if text.startswith("assistant"):
            text = text[len("assistant"):].lstrip("\n").strip()

        fence_json = re.search(r"```json\s*([\s\S]*?)```", text)
        if fence_json:
            text = fence_json.group(1).strip()
        else:
            fence_plain = re.search(r"```\s*([\s\S]*?)```", text)
            if fence_plain:
                text = fence_plain.group(1).strip()

        parsed = None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            pass

        if parsed is None:
            first = text.find("{")
            last = text.rfind("}")
            if first != -1 and last != -1 and last > first:
                try:
                    parsed = json.loads(text[first : last + 1])
                except json.JSONDecodeError:
                    pass

        if parsed is None:
            first = text.find("{")
            if first != -1:
                fragment = text[first:]
                try:
                    names_match = re.search(r'"names"\s*:\s*\[([^\]]*)', fragment)
                    emails_match = re.search(r'"emails"\s*:\s*\[([^\]]*)', fragment)
                    recovered_names = re.findall(r'"([^"]+)"', names_match.group(1)) if names_match else []
                    recovered_emails = re.findall(r'"([^"]+)"', emails_match.group(1)) if emails_match else []
                    parsed = {"names": recovered_names, "emails": recovered_emails}
                except Exception:
                    pass

        if parsed is None:
            self._parse_failure_log.append(
                {"idx": idx, "sequence": sentence, "raw_response": response}
            )
            return [], [], False

        names = [n for n in parsed.get("names", []) if isinstance(n, str) and n.strip()]
        emails = [e for e in parsed.get("emails", []) if isinstance(e, str) and e.strip()]

        sentence_lower = sentence.lower()
        names = [n for n in names if n.strip().lower() in sentence_lower]
        emails = [e for e in emails if e.strip().lower() in sentence_lower]

        return names, emails, True

    def _align_to_iob2(
        self, tokens: list[str], names: list[str], emails: list[str]
    ) -> list[str]:
        """Map extracted entity strings to IOB2 tags over the original token list.

        Emails are aligned first so their tokens are marked before name search.
        Names whose text is a substring of a parsed email are dropped to avoid
        double-tagging the local part of an address as both PER and EMAIL.
        """
        emails_lower = [e.strip().lower() for e in emails]
        names = [n for n in names if not any(n.strip().lower() in e for e in emails_lower)]

        tags = ["O"] * len(tokens)
        tokens_lower = [t.lower() for t in tokens]

        def _strip_punct(s: str) -> str:
            return s.strip(string.punctuation)

        def _find_and_tag(entity: str, b_tag: str, i_tag: str) -> None:
            if "@" in entity:
                ent_lower = entity.strip().lower()
                for i, tok in enumerate(tokens_lower):
                    if _strip_punct(tok) == _strip_punct(ent_lower) and tags[i] == "O":
                        tags[i] = b_tag
                        return
                parts = ent_lower.replace("@", " @ ").split()
                if len(parts) >= 2:
                    for start in range(len(tokens) - len(parts) + 1):
                        window = [_strip_punct(tokens_lower[start + k]) for k in range(len(parts))]
                        if window == [_strip_punct(p) for p in parts]:
                            if all(tags[start + k] == "O" for k in range(len(parts))):
                                tags[start] = b_tag
                                for k in range(1, len(parts)):
                                    tags[start + k] = i_tag
                                return

            ent_words = entity.strip().split()
            if not ent_words:
                return
            ent_words_lower = [w.lower() for w in ent_words]
            span_len = len(ent_words_lower)

            for start in range(len(tokens) - span_len + 1):
                window = [_strip_punct(tokens_lower[start + k]) for k in range(span_len)]
                if window == [_strip_punct(w) for w in ent_words_lower]:
                    if all(tags[start + k] == "O" for k in range(span_len)):
                        tags[start] = b_tag
                        for k in range(1, span_len):
                            tags[start + k] = i_tag
                        return

        for email in emails:
            _find_and_tag(email, "B-EMAIL", "I-EMAIL")
        for name in names:
            _find_and_tag(name, "B-PER", "I-PER")

        return tags

    def predict_sentence(self, tokens: list[str], sequence: str) -> dict:
        """Run inference on a single sentence and return a result record."""
        prompt = self._build_prompt(sequence)
        response = self.llm(
            prompt,
            max_tokens=256,
            temperature=_TEMPERATURE,
            stop=["<|eot_id|>", "<|end_of_text|>"],
        )
        raw_text = response["choices"][0]["text"]
        names, emails, parse_ok = self._parse_response(raw_text, sentence=sequence)
        predicted_tags = self._align_to_iob2(tokens, names, emails)

        return {
            "tokens": tokens,
            "sequence": sequence,
            "raw_response": raw_text,
            "parsed_names": names,
            "parsed_emails": emails,
            "predicted_tags": predicted_tags,
            "parse_ok": parse_ok,
        }

    def predict_batch(
        self,
        records: list[dict],
        cache_path: str,
        checkpoint_every: int = 50,
    ) -> list[dict]:
        """Run inference over a list of records with resume-from-cache support.

        Results are written atomically to cache_path every checkpoint_every
        new records so partial runs can be resumed without re-processing.
        """
        processed_indices: set[int] = set()
        cached_results: list[dict] = []
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        cached_results.append(obj)
                        if "idx" in obj:
                            processed_indices.add(obj["idx"])
                    except json.JSONDecodeError:
                        pass
            print(f"Resuming: {len(cached_results)} records already in cache.")

        results: list[dict] = list(cached_results)
        total = len(records)
        parse_failures = sum(1 for r in cached_results if not r.get("parse_ok", True))
        newly_processed = 0

        def _write_cache(path: str, data: list[dict]) -> None:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                for obj in data:
                    f.write(json.dumps(obj) + "\n")
            os.replace(tmp, path)

        for idx, record in enumerate(records):
            if idx in processed_indices:
                continue

            result = self.predict_sentence(record["tokens"], record["sequence"])
            result["idx"] = idx
            if "ner_tags" in record:
                result["ner_tags"] = record["ner_tags"]

            results.append(result)
            if not result["parse_ok"]:
                parse_failures += 1
            newly_processed += 1

            if newly_processed % 100 == 0:
                print(
                    f"Processed {idx + 1}/{total} sentences "
                    f"(parse failures so far: {parse_failures})"
                )

            if newly_processed % checkpoint_every == 0:
                _write_cache(cache_path, results)

        _write_cache(cache_path, results)
        print(
            f"Done. Processed {total} sentences total "
            f"(parse failures: {parse_failures})"
        )
        return results
