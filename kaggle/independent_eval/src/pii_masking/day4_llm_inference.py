"""Day 4 LLM prompting and output parsing pipeline."""

import json
import os
import re
import string
from typing import Optional

_parse_failure_log: list[dict] = []

SYSTEM_PROMPT = (
    "You are a precise PII detection system. Your task is to identify person names and email addresses\n"
    "in text. Return ONLY a valid JSON object. No explanations. No preamble. No markdown. No commentary.\n"
    "Rules:\n"
    "- Extract full names as they appear (e.g., \"John Smith\", not \"John\" and \"Smith\" separately)\n"
    "- Do NOT extract honorifics like Dr., Mr., Mrs., Prof. as part of the name unless inseparable\n"
    "- Do NOT extract organizations, locations, or other entities - only person names and emails\n"
    "- If no names or emails are found, return empty lists"
)


class LLMPIIPipeline:
    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int = 4, n_gpu_layers: int = 0):
        from llama_cpp import Llama
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads,
                         n_gpu_layers=n_gpu_layers, verbose=False)

    def _build_prompt(self, sentence: str) -> str:
        # Do NOT include <|begin_of_text|> - llama-cpp prepends it automatically.
        # Including it here causes a duplicate that degrades output quality.
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
        text = response.strip()

        # Strip spurious "assistant\n" prefix some llama-cpp versions emit
        if text.startswith("assistant"):
            text = text[len("assistant"):].lstrip("\n").strip()

        # Strip code fences
        fence_json = re.search(r"```json\s*([\s\S]*?)```", text)
        if fence_json:
            text = fence_json.group(1).strip()
        else:
            fence_plain = re.search(r"```\s*([\s\S]*?)```", text)
            if fence_plain:
                text = fence_plain.group(1).strip()

        parsed = None
        # Attempt 1: direct parse
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            pass

        # Attempt 2: extract first { ... last }
        if parsed is None:
            first = text.find("{")
            last = text.rfind("}")
            if first != -1 and last != -1 and last > first:
                try:
                    parsed = json.loads(text[first : last + 1])
                except json.JSONDecodeError:
                    pass

        # Attempt 3: truncation recovery - model was cut off mid-array or mid-string.
        # Use regex to extract only the complete quoted strings from each array,
        # bypassing any structural damage from truncation or looping.
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
            _parse_failure_log.append(
                {"idx": idx, "sequence": sentence, "raw_response": response}
            )
            return [], [], False

        names = [n for n in parsed.get("names", []) if isinstance(n, str) and n.strip()]
        emails = [e for e in parsed.get("emails", []) if isinstance(e, str) and e.strip()]

        # Hallucination filter: entity must appear in sentence (case-insensitive)
        sentence_lower = sentence.lower()
        names = [n for n in names if n.strip().lower() in sentence_lower]
        emails = [e for e in emails if e.strip().lower() in sentence_lower]

        return names, emails, True

    def _align_to_iob2(
        self, tokens: list[str], names: list[str], emails: list[str]
    ) -> list[str]:
        # Drop any name that is a substring of an email span - EMAIL wins.
        # Guards against the LLM redundantly listing "john.smith" in names
        # when "john.smith@acme.com" is already in emails. Without this,
        # a failed email token-match would leave the name tokens untagged
        # and they would then be picked up as B-PER / I-PER.
        emails_lower = [e.strip().lower() for e in emails]
        names = [n for n in names if not any(n.strip().lower() in e for e in emails_lower)]

        tags = ["O"] * len(tokens)
        tokens_lower = [t.lower() for t in tokens]

        def _strip_punct(s: str) -> str:
            return s.strip(string.punctuation)

        def _find_and_tag(entity: str, b_tag: str, i_tag: str) -> None:
            # Try single-token match for emails with @
            if "@" in entity:
                ent_lower = entity.strip().lower()
                for i, tok in enumerate(tokens_lower):
                    if _strip_punct(tok) == _strip_punct(ent_lower) and tags[i] == "O":
                        tags[i] = b_tag
                        return
                # Try split at @ as 2 tokens (user @ domain)
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

        # Emails first to prevent name overlap
        for email in emails:
            _find_and_tag(email, "B-EMAIL", "I-EMAIL")
        for name in names:
            _find_and_tag(name, "B-PER", "I-PER")

        return tags

    def predict_sentence(self, tokens: list[str], sequence: str) -> dict:
        prompt = self._build_prompt(sequence)
        response = self.llm(
            prompt,
            max_tokens=256,
            temperature=0.0,
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
        # Load already-processed indices from cache
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
