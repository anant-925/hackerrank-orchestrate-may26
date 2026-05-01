"""
Support Ticket Triaging Agent
==============================
Entry point for the HackerRank Orchestrate hackathon challenge.

Architecture:
  1. CorpusLoader   - loads all .md support docs from data/ directory
  2. TFIDFRetriever - finds top-k relevant docs per ticket query
  3. Classifier     - determines request_type, escalation need, product_area
  4. ResponseGenerator - produces grounded responses (Claude API or extraction fallback)
  5. Main pipeline  - reads support_tickets.csv, processes each row, writes output.csv

Usage:
    python main.py [--input PATH] [--output PATH] [--data-dir PATH]

Environment variables (optional):
    ANTHROPIC_API_KEY  - enables Claude API for high-quality response generation
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from dotenv import load_dotenv  # type: ignore[import]


def _load_env():
    """Load .env from project root if present."""
    repo_root = Path(__file__).resolve().parent.parent
    env_file = repo_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)


try:
    _load_env()
except ImportError:
    pass  # python-dotenv not installed; rely on shell environment

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

from corpus_loader import load_corpus
from retriever import TFIDFRetriever
from classifier import classify_request_type, should_escalate, infer_product_area
from response_generator import generate_response


DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_INPUT = Path(__file__).parent.parent / "support_tickets" / "support_tickets.csv"
DEFAULT_OUTPUT = Path(__file__).parent.parent / "support_tickets" / "output.csv"


def build_query(issue: str, subject: str, company: str) -> str:
    """Combine ticket fields into a single retrieval query, expanding key terms."""
    parts = []
    if subject and subject.strip() and subject.strip().lower() not in ("none", ""):
        parts.append(subject.strip())
    if issue and issue.strip():
        parts.append(issue.strip())
    if company and company.strip() and company.strip().lower() not in ("none", ""):
        parts.append(company.strip())

    query = " ".join(parts)

    # Expand common shorthand queries with better search terms
    query_lower = query.lower()
    expansions = []
    if "infosec" in query_lower or "security questionnaire" in query_lower:
        # "enhancing" matches the doc title "Enhancing your Account Security on Hackerrank for Work"
        expansions.append("account security enhancement hackerrank work vendor data")
    if "remove" in query_lower and "user" in query_lower:
        expansions.append("deactivate user remove member manage users lock access")
    if "subscription pause" in query_lower:
        expansions.append("pause subscription billing plan")
    if "rescheduling" in query_lower and "assessment" in query_lower:
        expansions.append("reschedule assessment test invite cancel")
    if "inactivity" in query_lower:
        expansions.append("inactivity timeout session interview settings")
    if "compatible check" in query_lower or ("zoom" in query_lower and "test" in query_lower):
        expansions.append("zoom audio video calls interviews connectivity requirements")
    if "resume builder" in query_lower or "creating resume" in query_lower:
        expansions.append("resume builder HackerRank profile")
    if "certificate" in query_lower and ("name" in query_lower or "update" in query_lower):
        expansions.append("certificate download share update name correction")
    if "employee" in query_lower and ("leaving" in query_lower or "left" in query_lower or "remove" in query_lower):
        expansions.append("deactivate user remove team member manage users lock access")
    if "dispute" in query_lower and "charge" in query_lower:
        expansions.append("dispute charge chargeback cardholder issuer billing statement")
    if "wrong product" in query_lower or ("merchant" in query_lower and "refund" in query_lower):
        expansions.append("dispute merchant transaction chargeback")
    if "minimum spend" in query_lower or ("minimum" in query_lower and "visa card" in query_lower):
        expansions.append("minimum transaction surcharge merchant rules interchange fees regulations")
    if "cash" in query_lower and ("urgent" in query_lower or "atm" in query_lower):
        expansions.append("emergency cash ATM Visa travel abroad")
    if "crawl" in query_lower or "robots" in query_lower:
        expansions.append("web crawl robots data training opt out")
    if "lti" in query_lower or ("student" in query_lower and "claude" in query_lower):
        expansions.append("LTI Canvas education university students Claude integration")
    if "bedrock" in query_lower or "aws" in query_lower:
        expansions.append("Amazon Bedrock AWS Claude contact support inquiries customer")
    if ("not responding" in query_lower or "stopped working" in query_lower or "all requests failing" in query_lower) and "claude" in query_lower:
        expansions.append("Claude API connection error troubleshoot failing service")
    if "mock interview" in query_lower or "mock interviews" in query_lower:
        expansions.append("mock interview subscription billing payment refund credits purchase")
    if "payment" in query_lower and ("order" in query_lower or "id" in query_lower):
        expansions.append("payment billing subscription order community HackerRank")
    if "submission" in query_lower and ("not working" in query_lower or "working" in query_lower):
        expansions.append("coding challenges practice submission community FAQ error")
    if "rescheduling" in query_lower and ("assessment" in query_lower or "test" in query_lower):
        # Candidate rescheduling requests - they need to contact the company
        expansions.append("reschedule test invite candidate company recruiter contact")

    if expansions:
        query = query + " " + " ".join(expansions)

    return query


def process_ticket(
    issue: str,
    subject: str,
    company: str,
    retriever: TFIDFRetriever,
) -> dict:
    """Process a single support ticket and return output fields."""
    issue = (issue or "").strip()
    subject = (subject or "").strip()
    company = (company or "").strip()

    # 1. Retrieve relevant docs
    query = build_query(issue, subject, company)
    company_filter = company if company.lower() not in ("none", "") else None
    retrieved = retriever.retrieve(query, company=company_filter, top_k=5)

    # 2. Check for escalation
    escalate, escalation_reason = should_escalate(issue, subject, company, retrieved)

    # 3. Classify request type
    request_type = classify_request_type(issue, subject, retrieved)

    # Override: invalid requests that can't/shouldn't be fulfilled get a canned response
    if request_type == "invalid" and not escalate:
        response = "I'm sorry, this request is outside the scope of what our support team can assist with."
        justification = "Request is invalid or not actionable by support (e.g., score change, out-of-domain query)."
        product_area = infer_product_area(retrieved, company, issue)
        return {
            "response": response,
            "product_area": product_area,
            "status": "replied",
            "request_type": request_type,
            "justification": justification,
        }

    # 4. Determine product area
    product_area = infer_product_area(retrieved, company, issue)

    # 5. Generate response
    response, justification = generate_response(
        issue, subject, company, retrieved,
        is_escalation=escalate,
        escalation_reason=escalation_reason,
    )

    status = "escalated" if escalate else "replied"

    return {
        "response": response,
        "product_area": product_area,
        "status": status,
        "request_type": request_type,
        "justification": justification,
    }


def run(input_path: str, output_path: str, data_dir: str):
    """Main pipeline: load corpus, process tickets, write output."""
    print(f"[1/4] Loading corpus from {data_dir} ...")
    corpus = load_corpus(data_dir)
    print(f"      Loaded {len(corpus)} documents.")

    print("[2/4] Building TF-IDF index ...")
    retriever = TFIDFRetriever(corpus)
    print("      Index ready.")

    print(f"[3/4] Processing tickets from {input_path} ...")
    rows = []
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    results = []
    for i, row in enumerate(rows, 1):
        issue = row.get("Issue", row.get("issue", ""))
        subject = row.get("Subject", row.get("subject", ""))
        company = row.get("Company", row.get("company", ""))
        print(f"  [{i:02d}/{len(rows)}] {subject or issue[:50]!r} ({company})")
        result = process_ticket(issue, subject, company, retriever)
        result.update({
            "issue": issue,
            "subject": subject,
            "company": company,
        })
        results.append(result)

    print(f"[4/4] Writing output to {output_path} ...")
    fieldnames = ["issue", "subject", "company", "response", "product_area", "status", "request_type", "justification"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone. Processed {len(results)} tickets → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Support ticket triaging agent")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input CSV path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Corpus data directory")
    args = parser.parse_args()

    run(args.input, args.output, args.data_dir)


if __name__ == "__main__":
    main()
