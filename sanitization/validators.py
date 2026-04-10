"""
Validation helpers to reduce false positives before masking.
"""
import re
import usaddress


def luhn_check(card_number: str) -> bool:
    """Return True only if card_number passes the Luhn algorithm."""
    digits = [int(d) for d in re.sub(r"\D", "", card_number)]
    if len(digits) < 13 or len(digits) > 19:
        return False
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    total = sum(odd_digits)
    for d in even_digits:
        total += sum(divmod(d * 2, 10))
    return total % 10 == 0


def is_real_address(text: str) -> bool:
    """
    Use usaddress to confirm the span looks like a postal address.
    Rejects law names, sentences, and narrative fragments.
    """
    noise_patterns = [
        r"\bact\b",
        r"\breform\b",
        r"\bprotection\b",
        r"\blaw\b",
        r"\bcompliance\b",
        r"\bproposed\b",
        r"\brequirement\b",
        r"\bpursuant\b",
        r"\bsection\b",
        r"\bthereof\b",
    ]
    lower = text.lower()
    for pat in noise_patterns:
        if re.search(pat, lower):
            return False
    try:
        tagged, _ = usaddress.tag(text)
        return "AddressNumber" in tagged or ("StreetName" in tagged and "PlaceName" in tagged)
    except Exception:
        return False


def is_real_party_name(text: str, max_token_length: int = 10) -> bool:
    """
    Filter out sentence fragments misclassified as party names.
    A real party name should be short and not contain sentence-like punctuation.
    """
    tokens = text.split()
    if len(tokens) > max_token_length:
        return False
    sentence_indicators = [
        r"\bbetween\b",
        r"\bwhich\b",
        r"\bthat\b",
        r"\bhave\b",
        r"\bwill\b",
        r"\bshall\b",
        r"\bmay\b",
        r"\bhas been\b",
        r"\bin order\b",
        r"\bas a\b",
    ]
    lower = text.lower()
    for pat in sentence_indicators:
        if re.search(pat, lower):
            return False
    return True


def is_real_financial_amount(text: str) -> bool:
    """
    Validate financial amounts. Must contain digits.
    Reject pure year sequences like '2016 2015 2014'.
    """
    clean = text.strip()
    if re.fullmatch(r"(\d{4}\s*)+", clean):
        return False
    return bool(re.search(r"\d", clean))


def is_real_ssn(text: str) -> bool:
    """Strict SSN format: NNN-NN-NNNN, no all-zero groups."""
    pattern = r"^\d{3}-\d{2}-\d{4}$"
    if not re.match(pattern, text):
        return False
    parts = text.split("-")
    if parts[0] == "000" or parts[1] == "00" or parts[2] == "0000":
        return False
    return True
