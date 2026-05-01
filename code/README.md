# Support Ticket Triaging Agent

A terminal-based AI agent that triages support tickets across three ecosystems (HackerRank, Claude, and Visa) using only the provided local support corpus.

## Architecture

```
support ticket
      │
      ▼
 build_query()          ← combines issue + subject + expansion keywords
      │
      ▼
 TFIDFRetriever         ← ranks 774 local support docs by cosine similarity
      │                    (2x boost for matching company; index files penalized)
      ▼
 should_escalate()      ← rule-based detection of high-risk patterns
 classify_request_type()←  request_type: product_issue / feature_request / bug / invalid
 infer_product_area()   ← derived from top doc's breadcrumbs/directory path
      │
      ▼
 generate_response()    ← Claude API (if ANTHROPIC_API_KEY set) → structured JSON
                          OR corpus extraction fallback (paragraph relevance scoring)
      │
      ▼
  output.csv
```

## Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point + query expansion + pipeline orchestration |
| `corpus_loader.py` | Loads + parses all `.md` files from `data/` |
| `retriever.py` | TF-IDF index (scikit-learn) with company boosting |
| `classifier.py` | Escalation rules, request_type, product_area logic |
| `response_generator.py` | LLM generation (Claude API) + corpus extraction fallback |

## Setup

```bash
# Install dependencies
pip install scikit-learn pandas anthropic python-dotenv

# Optional: set ANTHROPIC_API_KEY for LLM-backed responses
cp ../.env.example ../.env
# Edit .env and add your key
```

## Run

```bash
# From the code/ directory:
python main.py

# Custom paths:
python main.py --input ../support_tickets/support_tickets.csv \
               --output ../support_tickets/output.csv \
               --data-dir ../data
```

Output is written to `../support_tickets/output.csv`.

## Design Decisions

### Retrieval: TF-IDF over full corpus
- 774 support documents loaded from `data/hackerrank/`, `data/claude/`, `data/visa/`
- `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)` balances term frequency with rarity
- Documents from the matching company are given a **2× score boost**
- Index/navigation-only files (e.g. `index.md`) are **penalized 10×** to avoid returning table-of-contents as answers
- Query **expansion** enriches vague queries with domain-specific synonyms before retrieval

### Escalation Logic (rule-based)
Escalated when any of the following are detected:
- System-wide outages (`site is down`, `none of the pages accessible`)
- Identity theft or card fraud
- Prompt injection attempts — including French-language variants (`règles internes`, `logique exacte`)
- Legal / law enforcement requests
- Crisis/self-harm language
- Sensitive/destructive commands (`rm -rf`, `delete all files`, `sql inject`)
- Vague tickets with no company and insufficient context

### Response Generation
1. **Primary**: If `ANTHROPIC_API_KEY` is set, calls Claude API (`claude-3-5-haiku-20241022`) with retrieved docs as context and strict JSON output schema
2. **Fallback**: Extracts the most relevant paragraph(s) from retrieved docs using word overlap scoring with the original query

### Determinism
- TF-IDF is fully deterministic (no randomness)
- LLM calls use `temperature=0` for reproducibility

## Dependencies

```
scikit-learn>=1.0
pandas>=1.5
anthropic>=0.20  # optional, for LLM responses
python-dotenv    # optional
PyYAML
```
