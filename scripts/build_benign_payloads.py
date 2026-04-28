#!/usr/bin/env python3
"""
Build a diverse benign payload pool (~30k) for SQLi dataset construction.

Four sub-pools:
  1. pure_data           (~15k) — Faker-generated names, emails, products, ...
  2. special_chars       (~5k)  — apostrophes, percent signs, slashes, etc.
  3. attack_keyword_text (~5k)  — text that LOOKS attack-y but is benign content
                                   (tutorials, security logs, programmer chat)
  4. edge_case           (~5k)  — empty, whitespace, unicode, very long, ...

The 3rd pool is the critical one — it's what WAF-A-MoLE lacks and what
distinguishes "model learned semantics" from "model matched keywords".

Output: data/benign_payloads.json (same schema as attack_payloads.json)
"""
from __future__ import annotations
import json
import random
import string
from pathlib import Path

# Optional Faker — fall back to handcrafted if not installed
try:
    from faker import Faker
    HAVE_FAKER = True
except ImportError:
    HAVE_FAKER = False

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "benign_payloads.json"

random.seed(42)


# ============================================================
# 1. Pure data via Faker
# ============================================================
def gen_pure_data(n_total: int = 15000) -> list[dict]:
    out = []
    if not HAVE_FAKER:
        print("  WARNING: Faker not installed — skipping pure_data via Faker")
        # Fallback: use a small built-in pool
        names = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace",
                 "Henry", "Ivy", "Jack", "Kate", "Liam", "Mia", "Noah",
                 "Olivia", "Peter", "Quinn", "Rachel", "Sam", "Tina"]
        for i in range(n_total):
            out.append({
                "payload": random.choice(names) + " " + random.choice(names),
                "source": "fallback_names",
                "category": "pure_data",
            })
        return out

    fake = Faker(["en_US", "en_GB", "fr_FR", "de_DE"])
    Faker.seed(42)

    generators = [
        ("names",        2500, lambda: fake.name()),
        ("first_names",  500,  lambda: fake.first_name()),
        ("emails",       1500, lambda: fake.email()),
        ("addresses",    1500, lambda: fake.address().replace("\n", ", ")),
        ("phones",       500,  lambda: fake.phone_number()),
        ("companies",    800,  lambda: fake.company()),
        ("dates",        500,  lambda: fake.date()),
        ("urls",         500,  lambda: fake.url()),
        ("ids_uuid",     500,  lambda: fake.uuid4()),
        ("ids_short",    500,  lambda: ''.join(random.choices(string.ascii_uppercase + string.digits, k=random.randint(4, 12)))),
        ("integers",     500,  lambda: str(random.randint(0, 1_000_000))),
        ("floats",       300,  lambda: f"{random.uniform(-1000, 100000):.2f}"),
        ("countries",    300,  lambda: fake.country()),
        ("cities",       400,  lambda: fake.city()),
        ("words_short",  500,  lambda: " ".join(fake.words(nb=random.randint(1, 3)))),
        ("sentences",    3000, lambda: fake.sentence(nb_words=random.randint(4, 12))),
        ("currencies",   200,  lambda: fake.currency_code()),
        ("ipv4",         300,  lambda: fake.ipv4()),
        ("user_agents",  300,  lambda: fake.user_agent()[:200]),
        ("file_paths",   300,  lambda: fake.file_path()),
    ]

    for label, n, fn in generators:
        for _ in range(n):
            try:
                p = fn()
                if isinstance(p, str) and p.strip():
                    out.append({
                        "payload": p,
                        "source": f"faker_{label}",
                        "category": "pure_data",
                    })
            except Exception:
                continue
    return out


# ============================================================
# 2. Special chars (apostrophes, etc.)
# ============================================================
APOSTROPHE_NAMES = [
    "O'Brien", "O'Connor", "O'Reilly", "O'Hara", "O'Donnell", "O'Sullivan",
    "D'Angelo", "D'Souza", "D'Amore", "D'Arcy", "D'Onofrio",
    "L'Oreal", "L'Enfant",
    "McDonald's", "Wendy's", "Macy's", "Bobby's", "Tony's",
    "St. Mary's", "St. John's",
    "It's", "There's", "He's", "She's", "Won't", "Can't", "Don't",
    "Y'all", "I'm", "I'll", "I've", "Who's",
]

PUNCTUATED_DATA = [
    # Percent signs
    "100% cotton", "99.9% pure", "50% off", "25% discount",
    # Ampersands
    "Q&A session", "R&D team", "P&G products", "AT&T network",
    "M&Ms", "Ben & Jerry's",
    # Hyphens / dashes
    "Anne-Marie", "Jean-Paul", "Marie-Claire", "Mary-Kate",
    "non-fiction", "self-driving", "well-known", "long-term",
    # Slashes
    "5/5 stars", "and/or", "TCP/IP", "I/O port", "UTC/GMT",
    "MM/DD/YYYY", "kg/m^3",
    # Arrows / comparisons in text
    "5 > 3", "x <= 10", "a != b", "x => y",
    "if (x > 0)",
    # Markdown-like
    "**bold**", "*italic*", "`code`", "_underlined_", "~strikethrough~",
    # File paths
    "/var/log/syslog", "/etc/passwd",
    "C:\\Users\\test", "D:\\Projects\\app",
    # JSON / data
    '{"name":"Alice","age":30}',
    '{"key":"value","nested":{"inner":"data"}}',
    # CSV
    "Alice,30,USA",
    "id,name,email",
    # Mathematical
    "2 + 2 = 4", "x^2 + y^2 = z^2", "E = mc^2", "pi ≈ 3.14",
    # Currency
    "$100.00", "€50,99", "£75.50", "¥10000",
    # Quoted strings (escaped)
    'He said "hello"', '"Quote of the day"',
    # Time formats
    "12:00 PM", "23:59:59", "2024-01-15T10:30:00Z",
    # Phone formats
    "(555) 123-4567", "+1-202-555-0173",
    # Compound
    "Front-end + Back-end", "first; then second",
    "URL: https://example.com",
    "Email me @ user@example.com",
]


def gen_special_chars(n_total: int = 5000) -> list[dict]:
    out = []
    base = APOSTROPHE_NAMES + PUNCTUATED_DATA
    # Fill to n_total by sampling with replacement and adding a small suffix to vary
    for i in range(n_total):
        b = random.choice(base)
        # Sometimes prepend / append something to multiply variety
        op = random.random()
        if op < 0.4:
            payload = b
        elif op < 0.7:
            payload = b + " " + str(random.randint(1, 1000))
        elif op < 0.85:
            payload = "Mr. " + b if "'" in b else b + " Inc."
        else:
            payload = b
        out.append({
            "payload": payload,
            "source": "special_chars_curated",
            "category": "special_chars",
        })
    return out


# ============================================================
# 3. Attack-keyword-text (CRITICAL probe class)
# ============================================================
ATTACK_KEYWORDS = [
    "OR 1=1", "OR '1'='1'", "AND 1=1", "AND 0=0",
    "UNION SELECT", "UNION ALL SELECT", "SELECT *",
    "DROP TABLE", "DROP DATABASE", "DELETE FROM", "TRUNCATE",
    "INSERT INTO", "UPDATE SET",
    "SLEEP(5)", "BENCHMARK(", "WAITFOR DELAY",
    "EXTRACTVALUE", "UPDATEXML", "LOAD_FILE",
    "admin'--", "admin' #", "' OR '1'='1' --",
    "/**/comment/**/", "-- comment",
    "@@version", "USER()", "DATABASE()",
    "0x", "0xDEADBEEF", "CHAR(", "CONCAT(",
    "INTO OUTFILE", "INTO DUMPFILE",
    "information_schema",
    "1' AND ASCII",
]

PHRASES = [
    "How to prevent {kw} attacks",
    "Read about {kw} in security tutorials",
    "WAF blocked: {kw} attempt detected",
    "Use {kw} carefully in user inputs",
    "{kw} is a classic SQL injection pattern",
    "Tutorial: avoiding {kw} vulnerabilities",
    "What does '{kw}' mean in SQL injection context?",
    "Lecture notes on {kw}",
    "Blog post: dissecting {kw} payloads",
    "Pentest report mentions {kw}",
    "Audit log entry: detected {kw} signature",
    "Security alert: {kw} pattern in user input",
    "OWASP top 10 mentions {kw}",
    "Stack Overflow question about {kw}",
    "My friend told me about {kw} hack yesterday",
    "Don't paste {kw} into the production DB",
    "Avoid using {kw} in test fixtures",
    "Code review note: refactor away {kw} usage",
    "Email subject: Re: {kw} discussion",
    "Slack message: btw {kw} works on the staging env",
    "Bug report #1234: {kw} crashes the parser",
    "Remember: {kw} can be encoded as hex",
    "I tried {kw} but it didn't work",
    "The student wrote {kw} in the homework",
    "Conference talk: recent trends in {kw}",
    "Documentation page: {kw} examples and how to defend",
    "Whitepaper section 3.2 covers {kw}",
    "FAQ: what is {kw} and why should I care?",
    "Newsletter: this week we cover {kw}",
    "Wikipedia article: {kw}",
    "GitHub issue describing {kw}",
    "Reddit thread: {kw} explained",
    "Hacker News story mentions {kw}",
    "Read CVE-2024-XXXX details about {kw}",
    "Defcon talk on bypassing {kw} WAFs",
    "Internal training: {kw} in modern apps",
    "Q3 retrospective: zero {kw} incidents",
    "Hi team, please review this {kw} alert",
    "TIL: {kw} can also appear in NoSQL",
    "Test fixture comment: simulates {kw} input",
    "Penetration tester reported {kw} working on staging",
    "Backup note: don't include {kw} payloads in seed data",
]


def gen_attack_keyword_text(n_total: int = 5000) -> list[dict]:
    out = []
    for i in range(n_total):
        kw = random.choice(ATTACK_KEYWORDS)
        phrase = random.choice(PHRASES)
        payload = phrase.format(kw=kw)
        out.append({
            "payload": payload,
            "source": "attack_keyword_text_template",
            "category": "attack_keyword_text",
        })
    # Add some literal short attack-text-as-content (mimicking forum posts)
    LITERAL_PROBES = [
        "Is `{kw}` always malicious?",
        "I've seen `{kw}` in benign contexts",
        "Yo, what about {kw}? It got blocked.",
        "The string `{kw}` itself isn't an attack",
        "Searching for {kw} in our database",
        "Why did the WAF flag {kw}?",
    ]
    for i in range(500):
        kw = random.choice(ATTACK_KEYWORDS)
        phrase = random.choice(LITERAL_PROBES)
        out.append({
            "payload": phrase.format(kw=kw),
            "source": "attack_keyword_text_forum",
            "category": "attack_keyword_text",
        })
    return out


# ============================================================
# 4. Edge cases
# ============================================================
def gen_edge_cases(n_total: int = 5000) -> list[dict]:
    out = []
    # Short / boundary
    for _ in range(200):
        out.append({"payload": "", "source": "edge_empty", "category": "edge_case"})
    for c in " \t\n":
        for _ in range(100):
            out.append({"payload": c, "source": "edge_whitespace", "category": "edge_case"})
    for _ in range(500):
        out.append({"payload": random.choice(string.ascii_letters + string.digits + "@_-."),
                    "source": "edge_single_char", "category": "edge_case"})
    # Very long
    for _ in range(300):
        out.append({"payload": "x" * random.randint(200, 1000),
                    "source": "edge_long", "category": "edge_case"})
    # Numeric edge
    for v in ["0", "-0", "0.0", "1e10", "1e-10", "0xff", "Infinity", "NaN", "-1", "9999999999999"]:
        for _ in range(100):
            out.append({"payload": v, "source": "edge_numeric", "category": "edge_case"})
    # Boolean / null-like
    for v in ["true", "false", "null", "None", "TRUE", "FALSE", "NULL", "True"]:
        for _ in range(100):
            out.append({"payload": v, "source": "edge_boolish", "category": "edge_case"})
    # Unicode
    UNI_SAMPLES = [
        "中文测试字符串", "日本語のテキスト", "한국어 테스트", "العربية",
        "Hello 👋 World", "Café résumé naïve", "Привет мир", "Ñoño",
        "🎉🎊🎈", "Σ∑∫∂", "α + β = γ",
    ]
    for u in UNI_SAMPLES:
        for _ in range(80):
            out.append({"payload": u, "source": "edge_unicode", "category": "edge_case"})
    # HTML/XML escaped
    for h in ["&amp;", "&lt;script&gt;", "&#39;", "&quot;test&quot;", "<b>bold</b>"]:
        for _ in range(80):
            out.append({"payload": h, "source": "edge_html_escape", "category": "edge_case"})
    # URL with params
    for _ in range(300):
        out.append({"payload": f"https://example.com/path?id={random.randint(1, 1000)}&type={random.choice(['user','admin','guest'])}",
                    "source": "edge_url_with_params", "category": "edge_case"})
    # JSON snippets
    for _ in range(300):
        out.append({"payload": json.dumps({"id": random.randint(1, 100), "name": f"user_{random.randint(1, 1000)}", "active": random.choice([True, False])}),
                    "source": "edge_json", "category": "edge_case"})
    # CSV rows
    for _ in range(200):
        out.append({"payload": f"{random.randint(1, 1000)},user_{random.randint(1, 100)},active",
                    "source": "edge_csv", "category": "edge_case"})
    # Random base64-ish
    for _ in range(300):
        n = random.randint(20, 100)
        s = ''.join(random.choices(string.ascii_letters + string.digits + "+/=", k=n))
        out.append({"payload": s, "source": "edge_b64", "category": "edge_case"})
    # Pad to n_total
    while len(out) < n_total:
        out.append({"payload": str(random.randint(1, 1_000_000)),
                    "source": "edge_pad_int", "category": "edge_case"})
    return out[:n_total]


# ============================================================
# Main
# ============================================================
def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Generating pure_data...")
    pool_pure = gen_pure_data(15000)
    print(f"  {len(pool_pure)}")

    print("Generating special_chars...")
    pool_special = gen_special_chars(5000)
    print(f"  {len(pool_special)}")

    print("Generating attack_keyword_text...")
    pool_akt = gen_attack_keyword_text(5000)
    print(f"  {len(pool_akt)}")

    print("Generating edge_case...")
    pool_edge = gen_edge_cases(5000)
    print(f"  {len(pool_edge)}")

    all_payloads = pool_pure + pool_special + pool_akt + pool_edge

    # Add length and dedupe
    seen = set()
    unique = []
    for p in all_payloads:
        text = p["payload"]
        if text in seen:
            continue
        seen.add(text)
        p["length"] = len(text)
        unique.append(p)

    print(f"\nTotal before dedup: {len(all_payloads)}")
    print(f"Total after dedup:  {len(unique)}")

    # Stats
    from collections import Counter
    cat_counts = Counter(p["category"] for p in unique)
    src_counts = Counter(p["source"] for p in unique)
    print("\nBy category:")
    for cat, cnt in cat_counts.most_common():
        print(f"  {cat:30s} {cnt:>6}")
    print("\nTop sources:")
    for src, cnt in src_counts.most_common(15):
        print(f"  {src:30s} {cnt:>6}")
    lens = [p["length"] for p in unique]
    print(f"\nLength: min={min(lens)} median={sorted(lens)[len(lens)//2]} "
          f"p95={sorted(lens)[int(len(lens)*0.95)]} max={max(lens)}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(unique, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
