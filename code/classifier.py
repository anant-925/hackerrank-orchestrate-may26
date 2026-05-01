"""
Ticket Classifier: Determines request_type, escalation need, and product_area.
"""

import re
from typing import Dict, List, Optional, Tuple


# Patterns that trigger escalation (human handoff required)
ESCALATION_PATTERNS = [
    # System-wide outages
    r"\bsite (is )?down\b",
    r"\bnone of the (pages|services) are accessible\b",
    r"\bplatform (is )?down\b",
    r"\bcompletely (down|broken|unavailable)\b",
    # Financial / fraud
    r"\bidentity (theft|stolen)\b",
    r"\bcard (stolen|lost|compromised)\b",
    r"\bfraud\b",
    r"\bunauthori[sz]ed (charge|transaction|access)\b",
    # Requests to reveal internal info (prompt injection - English)
    r"\b(show|display|reveal|dump|print)\b.{0,30}\b(internal|system|config|rules|documents?|retrieved)\b",
    r"\binternal (rules|logic|documents)\b",
    r"\bexact(ly)? (what|how) you (use|decide)\b",
    # Prompt injection - French patterns
    r"\baffiche.{0,20}(r[eè]gles|documents|logique)\b",
    r"\br[eè]gles internes\b",
    r"\blogique exacte\b",
    r"\bdocuments r[eé]cup[eé]r[eé]s\b",
    # Legal / law enforcement
    r"\blaw enforcement\b",
    r"\bsubpoena\b",
    r"\blegal (request|demand)\b",
    # Crisis / urgent harm
    r"\bsuicid\b",
    r"\bself.harm\b",
    r"\bcrisis\b",
]

# Patterns for requests that are inherently unreasonable/invalid from support perspective
INVALID_REQUEST_PATTERNS = [
    # Candidates asking to change their test scores - support cannot do this
    r"\b(increase|change|update|fix|improve)\b.{0,30}\b(score|grade)\b.{0,30}\b(platform|hackerrank|unfair)\b",
    r"\btell.{0,30}(company|recruiter|employer).{0,30}(move|advance|hire|next round)\b",
    r"\bgraded.{0,20}unfairly\b",
]

# Patterns for bug classification
BUG_PATTERNS = [
    r"\bnot working\b",
    r"\bbroken\b",
    r"\b(can'?t|cannot|unable to) (see|access|use|open|submit|take|log in|login)\b",
    r"\berror\b",
    r"\bfailing\b",
    r"\bfailed\b",
    r"\bdown\b",
    r"\bissue with\b",
    r"\bblocker\b",
    r"\bbug\b",
    r"\bcrash\b",
    r"\bstop(ped)? working\b",
    r"\bnot (loading|responding|accessible|showing)\b",
    r"\bsubmissions?.*(not|fail)\b",
]

# Patterns for feature requests
FEATURE_REQUEST_PATTERNS = [
    r"\b(can you|could you|please) add\b",
    r"\b(would like|want) (a |to have |to see |to get )?(new |an? )?(feature|option|ability|way)\b",
    r"\bfeature request\b",
    r"\bwould be (nice|great|helpful) (if|to)\b",
    r"\bsuggest\b",
    r"\benhancement\b",
    r"\bwish\b",
]

# Patterns for invalid/out-of-scope tickets
INVALID_PATTERNS = [
    r"\bwho (is|was)\b.{0,30}\b(actor|actress|celebrity|singer|player)\b",
    r"\b(movie|film|song|music|sport)\b",
    r"\bwhat (is|are) the (name|capital|population)\b",
    r"\brecipe\b",
    r"\bweather\b",
    r"^(hi|hello|thanks?|thank you|ok|okay|yes|no)[.!?]?\s*$",
    r"^(great|good|nice|perfect|awesome)[.!?]?\s*$",
]

# Patterns that suggest sensitive/restricted operations
SENSITIVE_PATTERNS = [
    r"\bdelete all (files|data|everything)\b",
    r"\brm -rf\b",
    r"\bexecute\b.{0,20}\bcommand\b",
    r"\bsystem (command|call|exec)\b",
    r"\bsql inject\b",
    r"\bxss\b",
    r"\bhack\b",
    r"\bbypass\b.{0,20}\b(security|auth|check)\b",
    r"\b(steal|exfiltrate|leak)\b.{0,20}\b(data|info|credentials?)\b",
]


def _matches_any(text: str, patterns: List[str]) -> bool:
    """Check if text matches any pattern (case-insensitive)."""
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)


def classify_request_type(issue: str, subject: str, retrieved_docs: List[Dict]) -> str:
    """Determine the request_type: product_issue, feature_request, bug, invalid."""
    combined = f"{issue} {subject}".lower()

    if _matches_any(combined, INVALID_PATTERNS):
        return "invalid"

    # Requests that are inherently unreasonable (candidate asking to change scores etc.)
    if _matches_any(combined, INVALID_REQUEST_PATTERNS):
        return "invalid"

    if _matches_any(combined, FEATURE_REQUEST_PATTERNS):
        return "feature_request"
    if _matches_any(combined, BUG_PATTERNS):
        return "bug"
    return "product_issue"


def should_escalate(issue: str, subject: str, company: str, retrieved_docs: List[Dict]) -> Tuple[bool, str]:
    """
    Determine if the ticket needs human escalation.
    Returns (escalate: bool, reason: str).
    """
    combined = f"{issue} {subject}".lower()

    # Prompt injection / attempts to extract internal info
    if _matches_any(combined, SENSITIVE_PATTERNS):
        return True, "Potentially malicious or out-of-scope request detected."

    if _matches_any(combined, ESCALATION_PATTERNS):
        # Check specific cases
        if re.search(r"\bsite (is )?down\b|\bnone of the pages\b|\bplatform (is )?down\b", combined):
            return True, "System-wide outage reported. Requires immediate human investigation."
        if re.search(r"\bidentity (theft|stolen)\b", combined):
            return True, "Identity theft reported. Escalating to specialized support team."
        if re.search(r"\bfraud\b", combined):
            return True, "Potential fraud case. Escalating for human review."
        if re.search(r"\b(internal|system).{0,20}(rules|logic|documents)\b|r[eè]gles internes|logique exacte|documents r[eé]cup[eé]r[eé]s|affiche.{0,20}(r[eè]gles|documents|logique)", combined):
            return True, "Request to reveal internal system information detected. Potential prompt injection."
        if re.search(r"\bsuicid\b|\bself.harm\b|\bcrisis\b", combined):
            return True, "Crisis situation detected. Escalating with priority to support team."
        if re.search(r"\blaw enforcement\b|\bsubpoena\b", combined):
            return True, "Legal/law enforcement request. Requires legal team review."

    # If no corpus match at all and company is None or unclear - might need escalation
    if not retrieved_docs or (retrieved_docs and retrieved_docs[0].get("score", 0) < 0.05):
        if company in (None, "None", ""):
            return True, "Unable to determine relevant domain. Escalating for human review."

    # Vague ticket with no company and too little context to resolve
    combined_stripped = combined.strip()
    if company in (None, "None", "") and len(combined_stripped.split()) < 10:
        return True, "Insufficient context to route and resolve ticket. Escalating to human agent."

    return False, ""


def infer_product_area(retrieved_docs: List[Dict], company: str, issue: str) -> str:
    """
    Derive product_area from the best matching retrieved documents.
    """
    if not retrieved_docs:
        return _fallback_product_area(company, issue)

    # Use the top doc's product_area
    top_doc = retrieved_docs[0]
    area = top_doc.get("product_area", "")

    # Clean up the area
    area = area.strip("_").replace("__", "_")

    if area:
        return area

    return _fallback_product_area(company, issue)


def _fallback_product_area(company: str, issue: str) -> str:
    """Fallback product area based on company and issue keywords."""
    company_lower = (company or "").lower()
    issue_lower = issue.lower()

    if "hackerrank" in company_lower:
        if any(k in issue_lower for k in ["test", "assessment", "interview", "score", "invite"]):
            return "screen"
        if any(k in issue_lower for k in ["community", "profile", "account", "delete"]):
            return "community"
        if any(k in issue_lower for k in ["billing", "payment", "subscription", "refund"]):
            return "general_help"
        return "general_help"
    elif "claude" in company_lower:
        if any(k in issue_lower for k in ["conversation", "chat", "delete", "message"]):
            return "conversation_management"
        if any(k in issue_lower for k in ["billing", "payment", "subscription", "plan"]):
            return "billing"
        if any(k in issue_lower for k in ["privacy", "data", "personal"]):
            return "privacy"
        if any(k in issue_lower for k in ["api", "key", "bedrock", "console"]):
            return "claude_api_and_console"
        return "general_support"
    elif "visa" in company_lower:
        if any(k in issue_lower for k in ["travel", "cheque", "lost", "stolen"]):
            return "travel_support"
        if any(k in issue_lower for k in ["dispute", "charge", "refund"]):
            return "dispute_resolution"
        return "general_support"
    return "general_support"
