"""Day 1 synthetic email generation and injection utilities."""

import copy
import logging
import re

import numpy as np

logger = logging.getLogger(__name__)

EMAIL_REGEX = re.compile(r'^[a-z0-9._+-]+@[a-z0-9.-]+\.[a-z]{2,}$')
PUNCT_TOKENS = {".", "!", "?", ";"}


class EmailGenerator:
    """Generates synthetic email addresses from name information and config patterns."""

    def __init__(self, config: dict, split: str = "train") -> None:
        """Initialise generator for the given split section of config.

        Seeds numpy.random with config[split]["seed"] for reproducibility.
        """
        self.cfg = config[split]
        self.personal_domains: list[str] = self.cfg["personal_domains"]
        self.corporate_domains: list[str] = self.cfg["corporate_domains"]
        assert self.personal_domains, "personal_domains must not be empty"
        assert self.corporate_domains, "corporate_domains must not be empty"
        self.patterns: list[dict] = self.cfg["patterns"]
        self.role_patterns: list[dict] = self.cfg["role_patterns"]
        self.role_fraction: float = self.cfg["role_pattern_fraction"]
        # Track last chosen pattern for reporting
        self._last_pattern_name: str = ""
        # Seed numpy global state for generate_* methods
        np.random.seed(self.cfg["seed"])

    def generate_local_part(self, first: str, last: str) -> str:
        """Generate the local part of an email from a person's name.

        Uses role patterns role_pattern_fraction of the time; otherwise picks a
        name-based pattern by weight.
        """
        if np.random.random() < self.role_fraction:
            weights = [rp["weight"] for rp in self.role_patterns]
            total = sum(weights)
            probs = [w / total for w in weights]
            chosen = self.role_patterns[int(np.random.choice(len(self.role_patterns), p=probs))]
            self._last_pattern_name = chosen["name"]
            return chosen["local"]

        weights = [p["weight"] for p in self.patterns]
        total = sum(weights)
        probs = [w / total for w in weights]
        chosen = self.patterns[int(np.random.choice(len(self.patterns), p=probs))]
        self._last_pattern_name = chosen["name"]
        tmpl = chosen["template"]

        local = tmpl
        local = local.replace("{first}", first.lower())
        local = local.replace("{last}", last.lower())
        local = local.replace("{f}", first[0].lower() if first else "x")
        local = local.replace("{l}", last[0].lower() if last else "x")
        local = local.replace("{year}", str(np.random.randint(1970, 2006)))
        local = local.replace("{n}", str(np.random.randint(1, 100)))
        return local

    def generate_domain(self) -> str:
        """Pick a domain: 70% personal, 30% corporate."""
        if np.random.random() < 0.7:
            return self.personal_domains[int(np.random.randint(len(self.personal_domains)))]
        return self.corporate_domains[int(np.random.randint(len(self.corporate_domains)))]

    def generate_email(self, first: str, last: str) -> str:
        """Generate a full email address as local@domain.

        Falls back to a safe address if the result fails regex validation.
        """
        local = self.generate_local_part(first, last)
        domain = self.generate_domain()
        email = f"{local}@{domain}"
        if not EMAIL_REGEX.match(email):
            fallback_first = (first[0] if first else "x").lower()
            fallback_last = last.lower() if last else "user"
            email = f"{fallback_first}{fallback_last}@{self.personal_domains[0]}"
        return email


def extract_per_spans(ner_tags: list[str]) -> list[tuple[int, int, str]]:
    """Return (start, end_exclusive, 'PER') tuples for every PER span in ner_tags."""
    spans: list[tuple[int, int, str]] = []
    i = 0
    while i < len(ner_tags):
        if ner_tags[i] == "B-PER":
            start = i
            i += 1
            while i < len(ner_tags) and ner_tags[i] == "I-PER":
                i += 1
            spans.append((start, i, "PER"))
        else:
            i += 1
    return spans


def extract_name_from_span(tokens: list[str], start: int, end: int) -> tuple[str, str]:
    """Heuristically extract (first, last) name strings from a token span."""
    span_len = end - start
    if span_len == 1:
        return tokens[start], tokens[start]
    if span_len == 2:
        return tokens[start], tokens[start + 1]
    return tokens[start], tokens[end - 1]


def build_email_tokens(email: str, multi_token: bool) -> tuple[list[str], list[str]]:
    """Convert an email string into token + tag lists.

    Single-token: [email] / [B-EMAIL].
    Multi-token: [local, '@', domain] / [B-EMAIL, I-EMAIL, I-EMAIL].
    """
    if not multi_token:
        return [email], ["B-EMAIL"]
    local_part, domain = email.split("@", 1)
    return [local_part, "@", domain], ["B-EMAIL", "I-EMAIL", "I-EMAIL"]


def _verify_bio(tokens: list[str], tags: list[str]) -> None:
    """Assert no orphan I-X tags exist (raises AssertionError on violation)."""
    for j, tag in enumerate(tags):
        if not tag.startswith("I-"):
            continue
        entity = tag[2:]
        prev = tags[j - 1] if j > 0 else "O"
        assert prev in (f"B-{entity}", f"I-{entity}"), (
            f"BIO violation at position {j}: {tag!r} after {prev!r} "
            f"(token={tokens[j]!r})"
        )


def insert_email_into_example(
    example: dict,
    email_tokens: list[str],
    email_tags: list[str],
    per_span: tuple[int, int, str],
    context: str,
    rng: np.random.Generator,
) -> dict:
    """Insert email tokens into an example according to the named context strategy.

    Returns a new deep-copied dict; input is never mutated.
    Adds 'injected_email' metadata key to the result.
    """
    result = copy.deepcopy(example)
    tokens: list[str] = result["tokens"]
    tags: list[str] = result["ner_tags"]
    span_start, span_end, _ = per_span

    email_str = "@".join(
        [email_tokens[0]] if len(email_tokens) == 1 else [email_tokens[0], email_tokens[2]]
    )
    multi_token = len(email_tokens) == 3

    if context == "post_name":
        insert_at = span_end
        insert_toks = list(email_tokens)
        insert_tags_part = list(email_tags)
        # Add comma separator unless next token is punctuation
        if span_end < len(tokens) and tokens[span_end] not in PUNCT_TOKENS:
            insert_toks = [","] + insert_toks
            insert_tags_part = ["O"] + insert_tags_part
        tokens[insert_at:insert_at] = insert_toks
        tags[insert_at:insert_at] = insert_tags_part
        b_email_idx = insert_at + (1 if insert_toks[0] == "," else 0)

    elif context == "parenthetical":
        insert_at = span_end
        insert_toks = ["("] + list(email_tokens) + [")"]
        insert_tags_part = ["O"] + list(email_tags) + ["O"]
        tokens[insert_at:insert_at] = insert_toks
        tags[insert_at:insert_at] = insert_tags_part
        b_email_idx = insert_at + 1

    elif context == "contact_tail":
        if tokens and tokens[-1] in PUNCT_TOKENS:
            final_punct_tok = tokens.pop()
            final_punct_tag = tags.pop()
            tokens.extend(["Email:"] + list(email_tokens) + [final_punct_tok])
            tags.extend(["O"] + list(email_tags) + [final_punct_tag])
        else:
            tokens.extend(["Email:"] + list(email_tokens))
            tags.extend(["O"] + list(email_tags))
        b_email_idx = len(tokens) - len(email_tokens) - (1 if tokens[-1] in PUNCT_TOKENS else 0) - (0)
        # Recalculate: Email: is at position -(len(email_tokens)+1) from tail (before optional punct)
        # Simpler: find last B-EMAIL
        b_email_idx = next(i for i in range(len(tags) - 1, -1, -1) if tags[i] == "B-EMAIL")
    else:
        raise ValueError(f"Unknown context: {context!r}")

    result["tokens"] = tokens
    result["ner_tags"] = tags

    _verify_bio(tokens, tags)

    result["injected_email"] = {
        "email": email_str if not multi_token else "@".join([email_tokens[0], email_tokens[2]]),
        "context": context,
        "multi_token": multi_token,
        "inserted_at": b_email_idx,
    }
    # Always store the full email string
    full_email = email_tokens[0] if not multi_token else f"{email_tokens[0]}@{email_tokens[2]}"
    result["injected_email"]["email"] = full_email

    return result


def inject_dataset(
    examples: list[dict],
    config: dict,
    split: str = "train",
    seed: int = 42,
) -> tuple[list[dict], dict]:
    """Inject synthetic emails into PER-containing examples at the configured rate.

    Returns (output_examples, injection_report).
    """
    cfg_split = config[split]
    generator = EmailGenerator(config, split)
    rng = np.random.default_rng(seed)

    candidates_idx = [
        i for i, e in enumerate(examples)
        if any(t == "B-PER" for t in e["ner_tags"])
    ]
    n_to_inject = int(len(candidates_idx) * cfg_split["injection_rate"])

    chosen_idx = set(
        rng.choice(candidates_idx, size=n_to_inject, replace=False).tolist()
    )

    output: list[dict] = list(examples)

    contexts_list: list[str] = cfg_split["insertion_contexts"]
    context_weights = np.array(cfg_split["context_weights"], dtype=float)
    context_probs = context_weights / context_weights.sum()
    multi_token_ratio: float = cfg_split["multi_token_ratio"]

    single_count = 0
    multi_count = 0
    context_distribution: dict[str, int] = {c: 0 for c in contexts_list}
    domain_distribution: dict[str, int] = {}
    pattern_distribution: dict[str, int] = {}

    for i in sorted(chosen_idx):
        ex = examples[i]
        spans = extract_per_spans(ex["ner_tags"])
        if not spans:
            continue
        span = spans[int(rng.integers(len(spans)))]
        first, last = extract_name_from_span(ex["tokens"], span[0], span[1])

        email = generator.generate_email(first, last)
        pattern_name = generator._last_pattern_name
        domain = email.split("@")[1]

        multi_token = bool(rng.random() < multi_token_ratio)
        email_toks, email_tag_list = build_email_tokens(email, multi_token)

        ctx_idx = int(rng.choice(len(contexts_list), p=context_probs))
        context = contexts_list[ctx_idx]

        new_ex = insert_email_into_example(ex, email_toks, email_tag_list, span, context, rng)
        output[i] = new_ex

        if multi_token:
            multi_count += 1
        else:
            single_count += 1
        context_distribution[context] = context_distribution.get(context, 0) + 1
        domain_distribution[domain] = domain_distribution.get(domain, 0) + 1
        pattern_distribution[pattern_name] = pattern_distribution.get(pattern_name, 0) + 1

    # Shuffle for reproducibility
    perm = rng.permutation(len(output))
    output = [output[i] for i in perm]

    actual_rate = n_to_inject / len(candidates_idx) if candidates_idx else 0.0
    report = {
        "split": split,
        "total_examples": len(examples),
        "candidates": len(candidates_idx),
        "injected": n_to_inject,
        "injection_rate_actual": actual_rate,
        "single_token_count": single_count,
        "multi_token_count": multi_count,
        "context_distribution": context_distribution,
        "email_domain_distribution": domain_distribution,
        "pattern_distribution": pattern_distribution,
    }

    logger.info(
        "Injected %d emails (single=%d multi=%d) into %s split",
        n_to_inject, single_count, multi_count, split,
    )
    return output, report
