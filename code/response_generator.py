"""
Response Generator: Produces grounded responses using Claude API or
falls back to corpus extraction.
"""

import os
import re
import json
from typing import List, Dict, Optional

# Tuning constants
MAX_CONTEXT_DOCS = 4           # Max docs to include in LLM context window
MAX_DOC_CONTENT_LENGTH = 800   # Characters per doc in LLM context
MAX_RESPONSE_BUILD_LENGTH = 600  # Target char limit when building multi-paragraph response
MAX_FALLBACK_RESPONSE_LENGTH = 800  # Hard cap on extracted fallback response


SYSTEM_PROMPT = """You are an intelligent support ticket triaging agent.

Your task is to analyze a user support ticket and produce a structured response using ONLY the provided support documents.

Follow these steps strictly:

STEP 1: Understand the query - identify the user's core issue.

STEP 2: Use ONLY the provided documents to answer. Do NOT use external knowledge or hallucinate.

STEP 3: Generate a clear, helpful, professional response grounded strictly in the provided documents.

STEP 4: If the documents do not contain enough information to answer, say: "I'm sorry, I don't have enough information to resolve this. Please contact our support team directly."

OUTPUT FORMAT (respond with valid JSON only, no markdown):
{
  "response": "<user-facing answer, professional tone, grounded in docs>",
  "justification": "<1-2 sentences explaining why this response was given and which doc area was used>",
  "confidence": 0.0
}

RULES:
- Do NOT fabricate steps, policies, or information not in the documents.
- Do NOT deviate from the JSON schema.
- Keep responses concise and actionable.
- Prefer precision over verbosity.
"""


def _is_navigation_content(text: str) -> bool:
    """Check if extracted text is just a navigation/index page (not helpful as a response)."""
    # If it looks like a table of contents or index
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 3:
        return True
    # Count lines that look like links/list items vs actual content
    link_like = sum(1 for l in lines if l.startswith("-") or l.startswith("*") or l.startswith("#"))
    if link_like / max(len(lines), 1) > 0.7:
        return True
    return False


def _extract_response_from_docs(docs: List[Dict], issue: str) -> str:
    """
    Extract the most relevant content from docs as a response.
    Used as fallback when no LLM is available.
    """
    if not docs:
        return "I'm sorry, I don't have enough information to resolve this. Please contact our support team directly."

    issue_words = set(re.findall(r"\b\w{4,}\b", issue.lower()))

    for doc in docs[:3]:
        content = doc.get("content", "")
        if not content or _is_navigation_content(content):
            continue

        # Try to find the most relevant paragraph
        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 50]

        if not paragraphs:
            continue

        # Score paragraphs by word overlap
        best_para = paragraphs[0]
        best_score = 0
        for para in paragraphs:
            para_words = set(re.findall(r"\b\w{4,}\b", para.lower()))
            score = len(issue_words & para_words)
            if score > best_score:
                best_score = score
                best_para = para

        # Return top paragraphs for context
        response_parts = [best_para]
        for para in paragraphs:
            if para != best_para and len("\n\n".join(response_parts)) < MAX_RESPONSE_BUILD_LENGTH:
                response_parts.append(para)
            if len(response_parts) >= 3:
                break

        result = "\n\n".join(response_parts)[:MAX_FALLBACK_RESPONSE_LENGTH]
        if len(result) > 100:
            return result

    return "I'm sorry, I don't have enough information to resolve this specific issue from our documentation. Please contact our support team directly for further assistance."


def _call_claude_api(issue: str, subject: str, company: str, docs: List[Dict]) -> Optional[Dict]:
    """Call Claude API to generate a grounded response."""
    try:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return None

        client = anthropic.Anthropic(api_key=api_key)

        # Build context from retrieved docs
        doc_context = ""
        for i, doc in enumerate(docs[:MAX_CONTEXT_DOCS], 1):
            doc_context += f"\n\n--- Document {i}: {doc['title']} ---\n{doc['content'][:MAX_DOC_CONTENT_LENGTH]}"

        user_message = f"""Support ticket:
Company: {company}
Subject: {subject}
Issue: {issue}

Provided support documents:{doc_context}

Using ONLY the documents above, generate the response JSON."""

        message = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        raw = message.content[0].text.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)

    except Exception as e:
        print(f"  [LLM warning] {e}")
        return None


def generate_response(
    issue: str,
    subject: str,
    company: str,
    retrieved_docs: List[Dict],
    is_escalation: bool,
    escalation_reason: str,
) -> tuple[str, str]:
    """
    Generate (response, justification) for a ticket.
    Tries Claude API first, falls back to extraction.
    """
    if is_escalation:
        response = (
            "This ticket has been escalated to our support team for urgent attention. "
            "A human agent will review your case and respond shortly. "
            "We apologize for the inconvenience."
        )
        justification = escalation_reason or "High-risk or sensitive case escalated to human support."
        return response, justification

    # Try LLM
    llm_result = _call_claude_api(issue, subject, company, retrieved_docs)
    if llm_result and llm_result.get("response"):
        return llm_result["response"], llm_result.get("justification", "")

    # Fallback: corpus extraction
    response = _extract_response_from_docs(retrieved_docs, issue)
    top_doc_title = retrieved_docs[0]["title"] if retrieved_docs else "no relevant document"
    justification = (
        f"Response derived from: '{top_doc_title}'. "
        f"Matched via TF-IDF retrieval from {company or 'general'} corpus."
    )
    return response, justification
