"""Masking pipeline for PII redaction."""

import logging
from typing import List

logger = logging.getLogger(__name__)

PLACEHOLDER_MAP = {
    "PER": "[NAME]",
    "EMAIL": "[EMAIL]",
}


class MaskingPipeline:
    """Replace PII spans in text with [NAME] / [EMAIL] placeholders."""

    def mask_text(self, sequence: str, tokens: List[str], pred_tags: List[str]) -> str:
        """Redact all PER and EMAIL spans in sequence using predicted IOB2 tags.

        Operates on the original sequence string via character-offset replacement.
        Multi-token spans become a single placeholder. Processes left-to-right;
        any span whose start falls inside an already-replaced region is skipped.
        """
        if len(pred_tags) != len(tokens):
            raise ValueError(
                f"pred_tags length ({len(pred_tags)}) != tokens length ({len(tokens)})"
            )

        spans: list[tuple[int, int, str]] = []
        i = 0
        while i < len(tokens):
            tag = pred_tags[i]
            if tag.startswith("B-"):
                etype = tag[2:]
                j = i + 1
                while j < len(tokens) and pred_tags[j] == f"I-{etype}":
                    j += 1
                spans.append((i, j, etype))
                i = j
            else:
                i += 1

        token_char_starts: list[int] = []
        token_char_ends: list[int] = []
        pos = 0
        for tok in tokens:
            idx = sequence.find(tok, pos)
            if idx == -1:
                logger.warning(
                    "Token %r not found in sequence from position %d, skipping", tok, pos
                )
                token_char_starts.append(-1)
                token_char_ends.append(-1)
            else:
                token_char_starts.append(idx)
                token_char_ends.append(idx + len(tok))
                pos = idx + len(tok)

        replacements: list[tuple[int, int, str]] = []
        processed_end = -1
        for t_start, t_end, etype in spans:
            if token_char_starts[t_start] == -1:
                continue
            char_start = token_char_starts[t_start]

            last_valid = t_end - 1
            while last_valid >= t_start and token_char_ends[last_valid] == -1:
                last_valid -= 1
            if last_valid < t_start:
                continue
            char_end = token_char_ends[last_valid]

            if char_start < processed_end:
                continue  # overlapping - skip

            placeholder = PLACEHOLDER_MAP.get(etype, f"[{etype}]")
            replacements.append((char_start, char_end, placeholder))
            processed_end = char_end

        # Apply replacements in reverse order to preserve earlier offsets
        result = sequence
        for char_start, char_end, placeholder in reversed(replacements):
            result = result[:char_start] + placeholder + result[char_end:]

        return result

    def compute_leak_rate(self, records: List[dict]) -> dict:
        """Check what fraction of records still expose a gold PII entity after masking.

        A record 'leaks' if any gold PER or EMAIL entity string (case-insensitive,
        stripped) appears as a substring in the masked output produced by pred_tags.

        Returns {"leak_rate", "leaked_count", "total_count", "reconstructed_sequence_count"}.
        """
        leaked = 0
        total = 0
        reconstructed = 0

        for record in records:
            tokens: list[str] = record.get("tokens", [])
            gold_tags: list[str] = record.get("gold_tags") or record.get("ner_tags", [])
            pred_tags: list[str] = record.get("pred_tags") or record.get("predicted_tags", [])
            sequence: str | None = record.get("sequence")

            if sequence is None:
                sequence = " ".join(tokens)
                reconstructed += 1

            gold_entities: list[str] = []
            i = 0
            while i < len(gold_tags):
                tag = gold_tags[i]
                if tag.startswith("B-"):
                    etype = tag[2:]
                    span_tokens = [tokens[i]] if i < len(tokens) else []
                    j = i + 1
                    while j < len(gold_tags) and gold_tags[j] == f"I-{etype}":
                        if j < len(tokens):
                            span_tokens.append(tokens[j])
                        j += 1
                    entity_str = " ".join(span_tokens).strip()
                    if entity_str:
                        gold_entities.append(entity_str)
                    i = j
                else:
                    i += 1

            n = len(gold_tags)
            pred_aligned = (list(pred_tags) + ["O"] * n)[:n]

            try:
                masked = self.mask_text(sequence, tokens, pred_aligned)
            except ValueError:
                masked = sequence

            total += 1
            masked_lower = masked.lower()
            if any(ent.lower() in masked_lower for ent in gold_entities):
                leaked += 1

        leak_rate = leaked / total if total > 0 else 0.0
        return {
            "leak_rate": leak_rate,
            "leaked_count": leaked,
            "total_count": total,
            "reconstructed_sequence_count": reconstructed,
        }

    def mask_batch(self, records: List[dict]) -> List[dict]:
        """Add 'masked_sequence' key to every record in-place; return same list."""
        for record in records:
            tokens: list[str] = record.get("tokens", [])
            pred_tags: list[str] = record.get("pred_tags") or record.get("predicted_tags", [])
            sequence: str = record.get("sequence") or " ".join(tokens)

            n = len(tokens)
            pred_aligned = (list(pred_tags) + ["O"] * n)[:n]

            try:
                record["masked_sequence"] = self.mask_text(sequence, tokens, pred_aligned)
            except ValueError as exc:
                logger.warning("mask_text failed for record: %s", exc)
                record["masked_sequence"] = sequence

        return records
