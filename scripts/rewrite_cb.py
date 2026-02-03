import re


ALLOWED_TYPES = (
    "feat|fix|chore|docs|refactor|perf|test|ci|build|revert"
)

FIRST_LINE_OK = re.compile(
    rf"^({ALLOWED_TYPES})(\([^\)]+\))?: .+ \| EN: .+",
    re.IGNORECASE,
)

CAPTURED = re.compile(
    rf"^(?P<type>{ALLOWED_TYPES})(?:\\((?P<scope>[^)]+)\\))?: (?P<subject>.*?) \\| EN: (?P<en>.*)",
    re.IGNORECASE,
)


def _normalize_message_text(text: str) -> str:
    lines = text.splitlines()
    subject = (lines[0] if lines else "").strip()
    body = "\n".join(lines[1:]).strip()

    m = CAPTURED.match(subject)
    if m:
        # Reconstruct to avoid duplicates and normalize spacing/casing
        t = m.group('type').lower()
        scope = m.group('scope')
        subj = m.group('subject').strip()
        en = m.group('en').strip()
        head = f"{t}"
        if scope:
            head += f"({scope})"
        head += f": {subj} | EN: {en}"
        if body:
            return head + "\n\n" + body
        return head

    # Default to chore when type is missing/unknown; preserve original subject
    header = f"chore: {subject} | EN: update"
    if body:
        return header + "\n\n" + body
    return header


def rewrite(message: bytes, commit) -> bytes:  # noqa: D401
    """Transform commit message to Chinese-first conventional format.

    - If first line already matches pattern, keep untouched.
    - Else: wrap as `chore: <original subject> | EN: update` and preserve body.
    """
    try:
        text = message.decode("utf-8", errors="ignore")
    except Exception:
        return message
    new_text = _normalize_message_text(text)
    return new_text.encode("utf-8")


