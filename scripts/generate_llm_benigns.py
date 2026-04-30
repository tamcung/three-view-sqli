#!/usr/bin/env python3
"""Generate high-difficulty benign samples (hard negatives).

Seven categories of synthetic-but-realistic text containing SQL keywords,
attack phrases, function names, and SQL-shaped fragments — all in
contexts where the user's intent is benign:

  A. so_github       — Stack-Overflow / GitHub-issue style questions
  B. security_blog   — security tutorial / OWASP / pentest report
  C. waf_log         — ModSecurity / Cloudflare / SIEM alert lines
  D. code_review     — inline PR review comments
  E. bug_ticket      — JIRA / Linear / GitHub bug report tickets
  F. sql_shaped      — natural-language sentences mimicking SQL syntax
  G. multilingual    — Chinese / Japanese / Korean lines with SQL terms

Each template carries 1-6 slots filled from rich vocabularies. Per template
~25 expansions are sampled, dedupe applied, target ~5000+ unique outputs.

Output: data/llm_benigns.json   list of {payload, subtype, category, source, id, length}
"""
from __future__ import annotations
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "llm_benigns.json"


# ============================================================
# Vocabulary lists
# ============================================================
SQL_KEYWORDS = [
    "SELECT", "FROM", "WHERE", "INSERT", "UPDATE", "DELETE", "JOIN",
    "INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL OUTER JOIN", "CROSS JOIN",
    "UNION", "UNION ALL", "GROUP BY", "ORDER BY", "HAVING", "LIMIT", "OFFSET",
    "AND", "OR", "NOT", "IN", "NOT IN", "LIKE", "NOT LIKE", "BETWEEN",
    "IS NULL", "IS NOT NULL", "EXISTS", "NOT EXISTS",
    "CASE WHEN", "DISTINCT", "AS", "ON", "USING",
    "DROP TABLE", "DROP INDEX", "DROP DATABASE", "TRUNCATE", "ALTER TABLE",
    "CREATE TABLE", "CREATE INDEX", "CREATE VIEW", "CREATE PROCEDURE",
    "BEGIN TRANSACTION", "COMMIT", "ROLLBACK", "SAVEPOINT",
    "GRANT", "REVOKE",
    "VARCHAR", "INTEGER", "TIMESTAMP", "BOOLEAN", "JSONB",
    "WITH RECURSIVE", "LATERAL", "OVER (PARTITION BY", "WINDOW",
    "ROW_NUMBER()", "RANK()", "DENSE_RANK()", "LAG(", "LEAD(", "NTILE(",
    "PRIMARY KEY", "FOREIGN KEY", "REFERENCES", "ON DELETE CASCADE",
    "DEFAULT", "AUTO_INCREMENT", "SERIAL", "UNSIGNED",
    "TRUE", "FALSE", "NULL",
]

DANGEROUS_FUNCS = [
    # Information disclosure
    "@@version", "VERSION()", "CURRENT_USER()", "USER()", "DATABASE()",
    "@@hostname", "@@datadir", "@@version_compile_os", "@@global.tx_isolation",
    "SCHEMA()", "SESSION_USER()", "SYSTEM_USER()",
    # File I/O
    "LOAD_FILE", "INTO OUTFILE", "INTO DUMPFILE",
    # OS command
    "xp_cmdshell", "sp_OACreate", "sp_OAMethod", "sys_eval", "sys_exec",
    # Time-based
    "BENCHMARK(1000000, MD5('A'))", "SLEEP(5)", "SLEEP(10)",
    "WAITFOR DELAY '0:0:5'", "pg_sleep(5)", "DBMS_LOCK.SLEEP(5)",
    "DBMS_PIPE.RECEIVE_MESSAGE",
    # Error-based
    "EXTRACTVALUE(1, 0x7e)", "UPDATEXML(1, 0x7e, 1)",
    "GTID_SUBSET(1, 1)", "FLOOR(RAND(0)*2)", "ST_LatFromGeoHash",
    # JSON / advanced
    "JSON_EXTRACT", "JSON_KEYS", "JSON_TABLE", "->>",
    # Schema introspection
    "INFORMATION_SCHEMA.TABLES", "INFORMATION_SCHEMA.COLUMNS",
    "INFORMATION_SCHEMA.SCHEMATA",
    "mysql.user", "mysql.db", "pg_catalog.pg_user", "pg_catalog.pg_tables",
    "sys.databases", "sysobjects",
    # Concat tricks
    "GROUP_CONCAT", "CONCAT_WS", "CHAR(0x53,0x45,0x4C)",
]

ATTACK_PHRASES = [
    # Classic boolean
    "' OR 1=1 --", "1=1", "OR '1'='1", "AND 1=2",
    "' OR ''='", "1' OR '1'='1",
    "admin'--", "admin' #", "admin'/*",
    # Union
    "UNION SELECT", "UNION ALL SELECT NULL,NULL,NULL",
    "' UNION SELECT username, password FROM users --",
    # Time-based
    "1' AND SLEEP(5) --", "1' AND BENCHMARK(1000000, MD5('A')) --",
    # Error-based
    "' AND EXTRACTVALUE(1, 0x7e) --",
    "' AND UPDATEXML(1, CONCAT(0x7e, version()), 1) --",
    # Stacked
    "; DROP TABLE users; --",
    "; INSERT INTO users VALUES ('h4ck3r', 'pass') --",
    # Comments
    "/*!50000 SELECT */", "/*! UNION */", "-- malicious comment",
    # Encodings
    "%27%20OR%201%3D1%20--",
    "0x27204f5220313d31",
    "%u0027 OR %u0031%u003d%u0031",
    "&#x27; OR 1=1 &#x2D;&#x2D;",
    # Real-world payloads
    "1' AND (SELECT * FROM (SELECT(SLEEP(5)))a) --",
    "'; EXEC xp_cmdshell('whoami') --",
    # Function abuse
    "LOAD_FILE('/etc/passwd')",
    "INTO OUTFILE '/var/www/shell.php'",
    "(SELECT password FROM users WHERE username='admin')",
]

USER_NAMES = [
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
    "Ivan", "Julia", "Kevin", "Linda", "Mike", "Nancy", "Oscar", "Pam",
    "@dev_kim", "@infra_sam", "@security_lead", "@oncall", "@sre_team",
    "u_8472", "u_admin", "yamaguchi", "sokolov", "nakamura", "garcia",
    "patrick.j", "qa_team", "release_bot", "@maintainer",
]

TICKET_IDS = [
    "#1234", "#9876", "#42", "#7331", "#100", "#10042",
    "JIRA-892", "JIRA-1054", "JIRA-2099", "PROJ-451",
    "BUG-7261", "BUG-991", "BUG-3001", "BUG-0",
    "SEC-114", "SEC-205", "SEC-998", "OPS-1142",
    "GH-12345", "GH-987", "PR-2341", "ISSUE-5567",
]

CVE_LIST = [
    "CVE-2014-3704", "CVE-2017-7494", "CVE-2019-7255", "CVE-2021-44228",
    "CVE-2022-3236", "CVE-2023-23752", "CVE-2024-27198", "CVE-2024-1234",
    "CVE-2023-42115", "CVE-2022-22965", "CVE-2021-26084", "CVE-2020-1472",
]

DB_PRODUCTS = [
    "MySQL 5.7", "MySQL 8.0", "MariaDB 10.6", "PostgreSQL 14",
    "PostgreSQL 16", "SQL Server 2019", "SQL Server 2022",
    "Oracle 19c", "Oracle 23c", "SQLite 3.40",
    "MongoDB 6.0", "Redis 7.0", "ClickHouse 23.8", "TiDB 7.5",
]

ORM_NAMES = [
    "Hibernate", "MyBatis", "MyBatis-Plus", "Sequelize", "Prisma",
    "SQLAlchemy", "JOOQ", "TypeORM", "Knex", "GORM",
    "Eloquent", "Active Record", "Doctrine ORM", "Diesel", "Entity Framework",
]

WAF_PRODUCTS = [
    "ModSecurity CRS 4.0", "AWS WAF", "Cloudflare WAF",
    "Imperva SecureSphere", "F5 BIG-IP ASM", "Akamai App Protect",
    "FortiWeb", "Barracuda WAF", "Citrix ADC",
]

ATTACK_TYPES = [
    "SQL injection", "blind SQL injection", "time-based blind SQL injection",
    "error-based SQL injection", "UNION-based SQL injection",
    "stacked queries injection", "second-order SQL injection",
    "boolean-based blind injection", "out-of-band SQL injection",
    "NoSQL injection", "ORM injection", "JSON injection",
]

SLOT_TYPES = [
    "username", "password", "search query", "comment", "user_id",
    "product_id", "order_id", "filter", "sort_by", "page", "category",
    "email", "review_text", "API token", "X-Forwarded-For header",
    "Cookie value", "JSON body field",
]

URLS = [
    "/api/users", "/api/v1/search", "/login.php", "/admin/panel",
    "/products?id=1", "/orders/list", "/api/v2/items",
    "/checkout/cart", "/profile/edit", "/comments/post",
]

IP_ADDRS = [
    "203.0.113.42", "198.51.100.7", "192.0.2.146", "10.0.42.99",
    "172.16.5.123", "203.0.113.221", "198.51.100.84", "203.0.113.5",
]

TIMESTAMPS = [
    "2024-03-15 14:23:51", "2024-08-22 09:14:08", "2025-01-04 22:51:39",
    "[Mon Apr 15 13:42:18 2024]", "2024/11/03 07:55:11",
    "2025-02-14T19:08:42Z", "Apr  9 17:33:21",
]

LOG_LEVELS = ["INFO", "WARN", "ERROR", "CRITICAL", "ALERT", "BLOCK"]
RULE_IDS = ["942100", "942110", "942150", "942200", "942260", "942280",
            "942300", "942330", "942370", "942410", "942500"]

ATTACKER_GOALS = [
    "extract the admin password", "enumerate database tables",
    "dump the user table", "read /etc/passwd", "execute arbitrary commands",
    "exfiltrate session cookies", "elevate privileges to DBA",
    "create a backdoor account", "delete all rows", "perform reconnaissance",
]


# ============================================================
# Templates per category
# ============================================================

# A. Stack Overflow / GitHub Issue style
TEMPLATES_SO_GITHUB = [
    "How can I prevent {ATK} from being injected via the {SLOT} parameter in {DB}?",
    "Question: my query `SELECT * FROM users WHERE id = {VAL}` is breaking when user inputs {ATK}. Why?",
    "I'm using {ORM} {DB} — do I still need to worry about {ATTACK_TYPE}?",
    "Stack trace: ParseError near `{ATK}` in raw SQL — where is the issue {N}?",
    "Edit: closed because the {KW} clause is the problem; thanks {USER}",
    "Why does my code crash on `{ATK}`? It works fine when I escape the apostrophe",
    "{TICKET}: validating that PR #{N} actually fixes the {ATTACK_TYPE} reported by {USER}",
    "Comment from {USER}: I think your filter is missing the `{KW}` case",
    "Reproduction steps: `curl -X POST {URL} -d '{SLOT}={ATK}'` and you see a 500",
    "[BUG] Crash when `{ATK}` appears in {SLOT} field — see attached stack trace",
    "Pull Request #{N}: switch from raw {KW} to parameterized query, closes #{N2}",
    "Found {N} occurrences of unsafe `{KW}` concatenation in src/ during code audit",
    "Hi maintainers, I noticed that supplying {ATK} as the {SLOT} returns the {KW} error directly to the client",
    "After upgrading from {DB} to {DB2}, the previous payload `{ATK}` no longer triggers — fingerprint changed?",
    "Question: does {ORM} automatically escape `{ATK}`-like input, or do I need to call `prepare()` manually?",
    "Forum reply: the issue isn't {KW}, it's that you're concatenating `{ATK_PREVIEW}` instead of binding parameters",
    "Note added by {USER} on {TIMESTAMP}: the WAF blocks `{ATK}` but lets `{ATK2}` through, suggesting incomplete coverage",
    "I read in the OWASP cheatsheet that `{ATK}` is a classic test, but my legacy app still parses it without erroring",
    "Top answer: replace your raw `{KW}` with prepared statements; here is the {ORM} snippet",
    "Issue title says it all: parameter `{SLOT}` accepts `{ATK}` and the response status is 200 instead of 400",
    "Recap from yesterday's standup: {USER} reported that {ATK} bypasses our regex blacklist for `{KW}`",
    "Discussion in #{TICKET}: should we move to {ORM}'s native escape or stay on the custom {KW}-aware sanitizer?",
    "Tag: this falls under {CVE} category — an attacker can craft `{ATK}` to {GOAL}",
    "Best answer accepted from {USER}: never trust client input, always parameterize the {KW} clause",
    "I tested with payload `{ATK}` and `{ATK2}`; only the second one triggered the {ATTACK_TYPE} signature",
    "Bounty offered for a clean reproduction of the {KW} error when input contains {ATK}; current status: {STATUS}",
    "GitHub Discussions thread: how does {DB} handle `{ATK}` differently from {DB2}? edge cases please",
    "PR review nit: rename test case `test_filter_{KW_LOWER}` to make the {ATTACK_TYPE} intent explicit",
    "I keep getting downvoted, but `{ATK}` IS a legitimate test string for fuzzing parameter `{SLOT}`",
    "tl;dr — the {ORM} ORM does prevent `{ATK}` injection, but only when you avoid the `raw()` escape hatch",
    "Wiki edit summary: added section about `{ATK}` and how {WAF} rule {RULE} blocks it",
    "Linked issue {TICKET2} confirms that `{ATK}` is the same root cause as the original report from {USER}",
    "Self answer: turns out my {KW} subquery had an unbound parameter, the apparent `{ATK}` was a red herring",
    "Comment from a maintainer: please escape `<`, `>`, and the `{ATK_PREVIEW}` sequence before display",
    "Reading the source of {ORM} {VERSION}, I see that `escape_string` does handle `{ATK_PREVIEW}` correctly",
    "Old thread but relevant: someone hit the same `{KW}` parser ambiguity with {ATK} years ago",
    "Closing this in favor of #{TICKET} which has more details on the same `{ATK}` reproduction",
    "TIL: {DB}'s `{KW}` parser is more permissive than I thought, accepts `{ATK}` without error",
    "After much debugging, the culprit was a stale {ORM} migration that left a raw `{KW}` query in production",
    "Discussion thread: should `{ATK}` be added to the integration test suite alongside the existing `{ATK2}` cases?",
]

# B. Security blog / tutorial / pentest writeup
TEMPLATES_SECURITY_BLOG = [
    "Understanding {ATTACK_TYPE}\n\nA classic example: `{ATK}` injected into the {SLOT} field can {GOAL}",
    "OWASP recommends parameterized queries to prevent attacks like `{ATK}` and `{ATK2}`",
    "Lab walkthrough: bypass the login by entering `{ATK}` in the {SLOT} field",
    "PortSwigger Academy Lab #{N}: demonstrate {ATTACK_TYPE} via the URL `{URL}?id=1{ATK}`",
    "Pen-test report excerpt: payload `{ATK}` was successful against {DB} {VERSION} on host {IP}",
    "## How {ATTACK_TYPE} works\n\nThe attacker submits `{ATK}` and the backend executes `SELECT * FROM users WHERE id = 1{ATK}`",
    "Mitigation guide: filter for `{KW}`, `{KW2}`, and `{ATK_PREVIEW}` patterns; better yet, use prepared statements",
    "{CVE}: a remote attacker can submit `{ATK}` via the {SLOT} parameter to {GOAL}",
    "Defcon talk recap: speaker showed how `{ATK}` bypasses {WAF} when combined with `{KW}`-encoded chars",
    "OWASP Top 10 entry on injection: example payload `{ATK}` against a vulnerable PHP form",
    "Incident response notes: blocked `{ATK}` from {IP} at {TIMESTAMP}, attributed to {ATTACK_TYPE} fuzzing",
    "Threat intel report: APT group X uses obfuscated `{ATK}` against vulnerable {DB} endpoints",
    "Black-box pentest finding: parameter `{SLOT}` is vulnerable; PoC: `curl '{URL}?{SLOT}={ATK}'`",
    "Read-team writeup: chained `{ATK}` with a path traversal at {URL} to read {GOAL}",
    "Capture-the-flag challenge solution: the password is extracted via `{ATK}` after enumerating the schema with `{KW}`",
    "Workshop slide: 'See how `{ATK}` produces a verbose error revealing `{KW}` table structure'",
    "Defensive coding cheatsheet: never concatenate {SLOT} directly; always use {ORM} bound params or prepared {KW}",
    "Tutorial: using {WAF} to block `{ATK}` via rule {RULE} (regex match on `{KW_LOWER}`)",
    "Threat report Q3: observed {N}% increase in `{ATK_PREVIEW}` style probes across {N2} customer endpoints",
    "Conference presentation abstract: novel {ATTACK_TYPE} bypass using `{ATK}` against {DB} 8.0.{NUM}",
    "Lab solution: combine `{ATK}` with `{KW}` to trigger second-order injection on stored {SLOT}",
    "Bug bounty disclosure (truncated): payload `{ATK}` returned a stack trace exposing `{KW}` schema",
    "Security blog: '{ATK}' — anatomy of a {ATTACK_TYPE} that hit {DB} {VERSION}",
    "Forensics: log shows `{ATK}` from {IP} at {TIMESTAMP}, matched signature for {ATTACK_TYPE}",
    "Write-up by {USER}: chained `{ATK}` with `{KW}` to escalate from {ROLE} to DBA",
    "OWASP page recommends rejecting any input matching `{ATK_PREVIEW}` patterns at the WAF layer",
    "Course module 4: students will craft `{ATK}` and observe how {DB} parses it differently from {DB2}",
    "Vulnerability assessment summary: out of {N} parameters tested, {N2} were susceptible to `{ATK}`",
    "Defensive technique: 'allow-list' approach to {SLOT} validation rejects anything not matching `{REGEX}`",
    "{CVE} reproducer: send `POST {URL}` with body `{SLOT}={ATK}` and observe a delayed response of {SECS} seconds",
    "Forum post on bugcrowd: the report describes `{ATK}` triggering a 5-second backend delay (classic time-based)",
    "Pentest engagement scope: test for `{ATK}`-style {ATTACK_TYPE} on {URL}, {URL2}, and the {SLOT} cookie",
    "Conference Q&A: 'is `{ATK}` still effective against modern {WAF}?' — Yes, with the {KW} bypass technique",
    "Course chapter 5: demonstrate boolean-based blind extraction with payload `{ATK}` against the lab DB",
]

# C. WAF / SIEM / alert log lines
TEMPLATES_WAF_LOG = [
    "[{TIMESTAMP}] {LEVEL} ModSecurity: rule {RULE} triggered, blocked `{ATK}` from {IP}",
    "{TIMESTAMP} BLOCKED SQLi: {ATK}  src={IP}  uri={URL}",
    "Cloudflare event {TICKET}: SQLI category, payload preview: '{ATK_PREVIEW}'",
    "AWS WAF rule SQLI_BODY blocked req from {IP}; fragment: {ATK_PREVIEW}",
    "{LEVEL} {TIMESTAMP} {WAF} - blocked '{ATK}' (rule {RULE}, {ATTACK_TYPE})",
    "{TIMESTAMP} | level={LEVEL} | rule_id={RULE} | match='{ATK}' | uri={URL} | client={IP}",
    "[ALERT] {TIMESTAMP} - {N} requests with `{ATK_PREVIEW}` in {SLOT} param within last {SECS}s",
    "Splunk notable: count={N} sourcetype=waf src_ip={IP} payload_fragment='{ATK_PREVIEW}' ",
    "Sigma rule match: '{ATK}' detected via filter on `request.uri` — see incident {TICKET}",
    "ELK alert: query `request.body:\"{ATK_PREVIEW}\"` returned {N} hits in the last hour",
    "Suricata rule SID:{RULE}: SQLi attempt — pattern `{ATK_PREVIEW}` on {URL}",
    "{TIMESTAMP} {WAF} action=block client_ip={IP} matched_payload=`{ATK}`",
    "EDR telemetry: process=app.exe pid={N} sql_string=`{ATK}` flagged as {ATTACK_TYPE}",
    "On-call summary: {WAF} blocked {N} attempts overlapping with `{ATK_PREVIEW}` over the last {SECS}s",
    "SOC ticket {TICKET}: please review {N} alerts containing `{ATK_PREVIEW}` from {IP} cluster",
    "audit_log: user={USER} action=query stmt='{ATK}' result=denied reason=`{KW}`-pattern-match",
    "Daily digest: top blocked payload was `{ATK_PREVIEW}` ({N} hits across {N2} clients)",
    "Anomaly detector: payload `{ATK}` is {N}x rarer than baseline, flagged for review",
    "Outbound observation: server replied with verbose `{KW}` error, possibly leaking schema for {DB}",
    "Vendor advisory feed: `{ATK_PREVIEW}` matches signature for {CVE} ({ATTACK_TYPE})",
    "WAF tuning report: rule {RULE} false-positive rate {PCT}% on legitimate `{KW}` queries",
    "Postgres slow query log: query containing `{ATK_PREVIEW}` ran for {SECS}s, killed by timeout",
    "Daily summary email - {DATE}\nTop SQLi payloads blocked:\n  {ATK} ({N} hits)\n  {ATK2} ({N2} hits)",
    "Fluentd pipeline alert: {N} log lines with `{KW}` keyword in user_input field",
    "GuardDuty finding: SQLi:WebAppRequest from {IP} payload `{ATK_PREVIEW}` against bucket {SLOT}",
    "Network IDS: {N} TCP flows containing `{KW}` keyword in HTTP body, severity {LEVEL}",
]

# D. Code review / PR comment
TEMPLATES_CODE_REVIEW = [
    "review nit: this `{KW}` clause concatenates the user-supplied {SLOT}, vulnerable to `{ATK}`",
    "PR feedback by {USER}: replace raw `{KW}` with parameterized binding; add a regression test for `{ATK}`",
    "// FIXME: {KW} concatenation here, see {TICKET}",
    "review request comment: please add unit tests covering `{ATK}` and `{ATK2}` as inputs",
    "Suggested change (line {N}): use `{ORM}.escape({SLOT})` instead of building `{KW}` strings manually",
    "blocking review: this construction allows `{ATK}` through; please use prepared statements",
    "approval pending: tests pass, but I want to see a fixture for `{ATK}` before merging",
    "checklist item ✅: validated that `{KW}` is parameterized and `{ATK}` no longer reaches the DB",
    "TODO from {USER}: refactor the `{KW}` builder to refuse strings matching `{ATK_PREVIEW}` regex",
    "Minor: rename `userInput` to `rawUserInput` to flag that it might contain `{ATK}`-like content",
    "request changes: the migration script uses `{KW}` raw — what if {SLOT} contains `{ATK}`?",
    "/* SECURITY: do NOT pass user input directly into {KW}, always use {ORM} bind. See {TICKET}. */",
    "comment on diff: did we test the case where someone supplies `{ATK}` as the {SLOT}?",
    "git blame on this line shows the `{KW}` concat was added in commit {SHA} — predates the {ORM} migration",
    "// NOTE: `{ATK}` reaching this point should be impossible — assert it isn't",
    "RFC review: section 3.{N} mentions input validation but doesn't address `{ATK}` style payloads",
    "Threading model concern: if {SLOT} can be `{ATK}` and we run `{KW}` async, we need to escape per-call",
    "Inline review on line 120: this regex `{REGEX}` won't catch encoded `{ATK_PREVIEW}` — extend or remove",
    "Approving with comments: the `{KW}` rewrite looks good but please add an integration test that submits `{ATK}`",
    "CODEOWNERS triggered review for security-sensitive `{KW}` change; @{USER} please confirm `{ATK}` cannot leak",
    "Discussion: should the linter flag any `{KW}` followed by string concat? caught {N} occurrences in last sweep",
    "Self-review: I left a TODO to handle `{ATK}` properly — splitting into a follow-up PR {TICKET}",
    "Reviewer reminder: legacy `{KW}` paths in src/legacy/ are still exploitable via `{ATK}`, low priority",
    "Pre-merge checklist incomplete: missing 'tested with `{ATK}` payload' confirmation on the {SLOT} field",
    "Inline concern: the `{KW}` query trusts `request.params.{SLOT}` directly — verify with `{ATK}` in stage",
    "Co-author thread: agreed to parameterize, but the legacy `{KW}` adapter still calls raw exec; punt to {TICKET}",
]

# E. Bug ticket / issue tracker
TEMPLATES_BUG_TICKET = [
    "{TICKET}: customer reported {ATK} crashes the parser when entered in the {SLOT} field",
    "Severity: {LEVEL}. Repro: visit {URL} and the {SLOT} parameter accepts `{ATK}` without sanitization",
    "Issue {TICKET}: investigating why a {KW} returns wrong results when filter is `{ATK}`",
    "Closed as duplicate of {TICKET2} ({ATTACK_TYPE} via {SLOT})",
    "Postmortem: deploy at {TIMESTAMP} broke when {USER} ran a {KW} statement on prod with input `{ATK}`",
    "Ticket {TICKET}: {USER} pasted `{ATK}` into the description by accident — no actual exploit, but logs filled",
    "Title: '{ATK} in {SLOT} field renders as 500 not 400'\nPriority: {LEVEL}",
    "Steps to reproduce:\n1. Open {URL}\n2. Type `{ATK}` in {SLOT}\n3. Press submit\n4. Observe 500",
    "Expected: form rejects `{ATK}` with friendly message. Actual: stack trace exposing `{KW}` schema",
    "Reopened: previous fix only handled `{ATK_PREVIEW}` exact match, not `{ATK2}` variants",
    "Linked CVE: {CVE} — same root cause as our {KW} concat issue",
    "Customer-facing impact: search box on {URL} accepts `{ATK}` and returns DB error to client",
    "Triage: {LEVEL}, assigned to {USER}, fix expected by {DATE}; affected version: {VERSION}",
    "RCA: the `{KW}` template was missing escape for `{ATK_PREVIEW}` apostrophe form",
    "Risk assessment: P{NUM}, {N} customers exposed, attacker would need {SLOT} field access plus `{ATK}` knowledge",
    "Workaround: temporarily reject any input matching `{ATK_PREVIEW}` regex on the WAF layer",
    "Reproduction screenshot attached: error page shows raw `{KW}` query with `{ATK}` interpolated",
    "Field testing: 4 of 5 staging endpoints pass, but {URL} still emits the {KW} verbose error on `{ATK}`",
    "Linked PR: #{N} adds the prepared-statement fix and a regression test asserting `{ATK}` is rejected",
    "Status update from {USER}: rolled out fix to {N} of {N2} regions; remaining will deploy by {DATE}",
    "Comment thread: SRE confirmed no exploitation observed in logs, but `{ATK}` was sent {N} times by automated scanners",
    "Internal note: SLO breach driven by retries on `{ATK}` payloads triggering 500s; throttling now in place",
    "Escalation summary: customer {USER} contacted support after `{ATK}` produced an unexpected 200 instead of 400",
    "Template field: 'expected behavior' → input like `{ATK}` should be sanitized before reaching the {KW} layer",
    "Reopened by QA: the fix in {TICKET2} addresses `{ATK}` but `{ATK2}` (the URL-encoded form) still reaches the parser",
]

# F. SQL-shaped natural language (semantic puns)
TEMPLATES_SQL_SHAPED = [
    "SELECT your favorite color FROM red, blue, green WHERE you are happy",
    "INSERT INTO conversation VALUES ('hello world')",
    "UPDATE feelings SET mood = 'better' WHERE day = 'Monday'",
    "DELETE all worries FROM your mind",
    "DROP everything and read this email NOW",
    "FROM the bottom of my heart, thank you",
    "WHERE there is a will, there is a way",
    "JOIN us at 5pm in the meeting room",
    "ORDER BY priority — call mom first",
    "SELECT all items in your cart and proceed to checkout",
    "INSERT yourself INTO the conversation gracefully",
    "GROUP BY your strengths, conquer your weaknesses",
    "HAVING a good day requires effort",
    "WHERE were you when the lights went out?",
    "UNION of two souls is called marriage",
    "TRUNCATE your sleep and you'll feel terrible",
    "ALTER your perspective and the world changes",
    "CREATE space for what matters",
    "BEGIN today with intention",
    "COMMIT to the process, not the outcome",
    "ROLLBACK to the version that worked",
    "SELECT happiness AS default state FROM life",
    "EXISTS a bug in production — find it now",
    "Search: SELECT user FROM directory WHERE name LIKE '%john%'",
    "Form field: enter your password (8 chars min, no special)",
    "Quiz Q{N}: what does WHERE clause do? a) filter b) sort",
    "Cheat sheet: 'SELECT * FROM table' returns all rows",
    "Mnemonic: SELECT * FROM users WHERE active = TRUE",
    "Slide: '({KW}) ← THIS IS WHERE THE BUG LIVES'",
    "Note to self: SELECT good habits FROM today onwards",
    "SELECT * FROM the menu, please",
    "Today's special: UPDATE pasta SET sauce = 'tomato'",
    "WHERE id = 1 means \"the chosen one\"",
    "INSERT 'thanks' INTO every email reply",
    "SELECT 1 FROM dual is the simplest health check",
    "(WHERE 1=1) is a neutral filter, often used in dynamic SQL",
    "1=1 is trivially true — used as default WHERE in MyBatis dynamic SQL",
    "Drop the bass, not the tables",
    "Insert keys into the lock, turn clockwise",
    "SELECT a winner from the list of nominees",
    "I had to UPDATE my resume after the layoff",
    "Don't TRUNCATE the customer's name when displaying",
    "Where there's smoke, there's fire",
    "When in Rome, do as Romans do",
    "All for one and one for all",
    "From dawn to dusk, the world keeps turning",
    "Order from chaos — that's the engineer's job",
    "Group hug, everyone made it through Q{N}",
    "Insert coin to continue",
    "Update on the situation: still no internet",
    "Delete that text I sent — wrong chat",
    "Drop me a line when you're free",
    "Truncate your speech to under {N} minutes",
    "Alter the timeline if you can",
    "Court orders DROP TABLE on illegal data",
    "Senator calls for UPDATE on infrastructure bill",
    "From Russia With Love (1963) — classic Bond",
    "DELETE button removed from latest UI redesign",
    "WHERE IS THE LOVE — Black Eyed Peas",
    "She has a Ph.D. in databases and specializes in JOIN optimization",
    "The DELETE key on my keyboard is broken",
    "His last name is O'Brien — yes, with the apostrophe",
    "Café-au-lait, please. Tea? UNION pls",
    "if (user == 'admin') { return TRUE; }",
    "for x in range({N}): print(x)",
    "git commit -m 'fix UPDATE bug'",
    "git rebase -i HEAD~{N}",
    "docker run --rm postgres:14",
]

# G. Multilingual (Chinese / Japanese / Korean / mixed)
TEMPLATES_MULTILINGUAL = [
    "Bug 报告 {TICKET}：用户在 {SLOT} 字段输入 {ATK} 后后端崩溃",
    "教程：在 PHP 里如何安全使用 {KW} 子句",
    "代码审查：这段 {KW} 拼接有 SQL 注入风险，请用 {ORM} 参数化",
    "工单 {TICKET}：备注字段里出现了 `{ATK}`，看起来是误操作但触发了 500",
    "讨论：`{FUNC}` 在我们的项目里有合法用途吗？",
    "性能问题：`{KW}` 查询在 {N} 万行上跑得太慢，建议加索引",
    "用户在搜索框输入了 `{ATK}`，被 WAF 拦截",
    "数据库管理员小心使用 `{FUNC}` 函数",
    "本周共检测到 {N} 次包含 {KW} 的可疑请求",
    "请勿在生产环境直接执行 `{KW}` 语句",
    "工程师报告：`{KW}` 子句的语义在新版本中有变化",
    "审计日志显示用户 {USER} 于 {TIMESTAMP} 试图执行 `{KW}` 操作",
    "我们决定 `DROP` 掉那个老旧模块",
    "请提交 `UPDATE` 后的设计文档",
    "数据库审计：本月 `SELECT` 查询超过 {N} 亿次",
    "今天小明在群里发了一句 `' OR 1=1`，被同事当成段子",
    "Anti-SQLi 培训第三章：`UNION SELECT` 攻击防御",
    "WAF 报告：`{ATK}` 这条规则今日命中 {N} 次",
    "渗透测试报告：发现 `{SLOT}` 参数对 `{ATK}` 易受 {ATTACK_TYPE} 攻击",
    "运维群里讨论：`{FUNC}` 在 {DB} 上的执行计划怎么调",
    "新员工入职培训 PPT 第 {N} 页：`{ATK}` 是经典 SQLi 范例",
    "钉钉机器人推送：检测到 `{ATK}` 来自 {IP}，已自动封禁",
    "[Slack #security-alerts] {USER} 报了一个 `{KW}` 漏洞，影响范围尚在评估",
    "Code review 上 {USER} 指出：这里的 `{KW}` 拼接没做转义，提交了 PR #{N}",
    "メモ: ユーザー入力 `{ATK}` が DB の `{KW}` クエリに直接注入される可能性あり",
    "セキュリティレビュー: `{ATK}` を `{SLOT}` パラメータに渡すとエラーが発生",
    "コードレビュー: `{KW}` を使う前にプリペアドステートメントへ移行",
    "버그 #{N}: 사용자가 `{SLOT}` 필드에 `{ATK}` 를 입력하면 서버가 응답하지 않음",
    "교육 자료: `{KW}` 절에 사용자 입력을 직접 결합하는 것은 위험합니다",
    "보안 공지: `{ATK}` 패턴이 포함된 트래픽 차단 정책 적용",
    "运营周报：本周拦截 `{ATK_PREVIEW}` 类请求 {N} 次，已通报开发组",
    "测试用例 {TICKET}：模拟攻击者在 `{SLOT}` 输入 `{ATK}`，期望返回 400",
    "翻译说明：'inject' 在 SQL 注入语境下译作'注入'，与 `{KW}` 关键字区分",
    "群里同事问：`{ATK}` 这种语句到底是怎么绕过 WAF 的？以下是分析",
    "客户反馈：填写产品名时不小心输入了 `{ATK}`，请问会不会被记录到日志？",
    "{KW} 子查询写错容易报 `{FUNC}` 错误，建议先在测试库跑一遍",
    "运维提醒：勿在生产 `{KW}` 中使用 `{ATK_PREVIEW}` 这种字面量做调试",
    "面试题：解释为什么 `{ATK}` 是经典 {ATTACK_TYPE} 示例",
    "技术分享：从一次真实的 `{ATK}` 攻击复盘 {DB} 的输入校验薄弱点",
]


CATEGORIES = {
    "so_github":      ("llm_so_github",      TEMPLATES_SO_GITHUB),
    "security_blog":  ("llm_security_blog",  TEMPLATES_SECURITY_BLOG),
    "waf_log":        ("llm_waf_log",        TEMPLATES_WAF_LOG),
    "code_review":    ("llm_code_review",    TEMPLATES_CODE_REVIEW),
    "bug_ticket":     ("llm_bug_ticket",     TEMPLATES_BUG_TICKET),
    "sql_shaped":     ("llm_sql_shaped",     TEMPLATES_SQL_SHAPED),
    "multilingual":   ("llm_multilingual",   TEMPLATES_MULTILINGUAL),
}


# ============================================================
# Slot expansion
# ============================================================
SHA_SAMPLES = ["a3f9c2b", "1e7dd0a", "f0c4912", "789ab12", "deadbee", "cafef00",
                "33b1c1f", "0099fa3", "feedfac", "c0ffee5"]
DATES = ["2024-04-15", "2024-09-01", "2024-12-30", "2025-01-15", "2025-03-08",
          "2025-06-22", "2024-07-11", "2024-11-20", "2025-02-04", "2025-08-19"]
ROLES = ["read-only", "viewer", "regular user", "moderator", "support agent", "anonymous"]
SECS = ["3", "5", "10", "30", "60", "120"]
PCTS = ["0.5", "1.2", "2.8", "4.1", "12.5", "23"]
NUMS = ["123", "999", "5", "1024", "50", "8", "3", "100", "404", "500", "42"]
VERSIONS = ["1.0.5", "2.3.7", "3.14.2", "5.0.0", "7.4.1", "11.2.3"]
REGEXES = [r"^[a-z0-9_]+$", r"\\b{1,20}\\b", r"^\\d{1,10}$", r"[A-Za-z]+@[A-Za-z]+"]


def fill(template: str, rng: random.Random) -> str:
    sub_atk = rng.choice(ATTACK_PHRASES)
    sub_atk2 = rng.choice(ATTACK_PHRASES)
    while sub_atk2 == sub_atk:
        sub_atk2 = rng.choice(ATTACK_PHRASES)

    text = template
    replacements = {
        "{KW}": rng.choice(SQL_KEYWORDS),
        "{KW2}": rng.choice(SQL_KEYWORDS),
        "{KW_LOWER}": rng.choice(SQL_KEYWORDS).lower().replace(" ", "_"),
        "{FUNC}": rng.choice(DANGEROUS_FUNCS),
        "{ATK}": sub_atk,
        "{ATK2}": sub_atk2,
        "{ATK_PREVIEW}": sub_atk[:30] + ("..." if len(sub_atk) > 30 else ""),
        "{USER}": rng.choice(USER_NAMES),
        "{TICKET}": rng.choice(TICKET_IDS),
        "{TICKET2}": rng.choice(TICKET_IDS),
        "{CVE}": rng.choice(CVE_LIST),
        "{DB}": rng.choice(DB_PRODUCTS),
        "{DB2}": rng.choice(DB_PRODUCTS),
        "{ORM}": rng.choice(ORM_NAMES),
        "{WAF}": rng.choice(WAF_PRODUCTS),
        "{ATTACK_TYPE}": rng.choice(ATTACK_TYPES),
        "{SLOT}": rng.choice(SLOT_TYPES),
        "{URL}": rng.choice(URLS),
        "{URL2}": rng.choice(URLS),
        "{IP}": rng.choice(IP_ADDRS),
        "{TIMESTAMP}": rng.choice(TIMESTAMPS),
        "{LEVEL}": rng.choice(LOG_LEVELS),
        "{RULE}": rng.choice(RULE_IDS),
        "{GOAL}": rng.choice(ATTACKER_GOALS),
        "{N}": str(rng.randint(2, 9999)),
        "{N2}": str(rng.randint(2, 999)),
        "{NUM}": rng.choice(NUMS),
        "{VAL}": str(rng.randint(1, 9999)),
        "{SHA}": rng.choice(SHA_SAMPLES),
        "{DATE}": rng.choice(DATES),
        "{ROLE}": rng.choice(ROLES),
        "{SECS}": rng.choice(SECS),
        "{PCT}": rng.choice(PCTS),
        "{VERSION}": rng.choice(VERSIONS),
        "{STATUS}": rng.choice(["open", "closed", "wontfix", "duplicate", "in-progress"]),
        "{REGEX}": rng.choice(REGEXES),
    }
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    return text


def main():
    rng = random.Random(42)

    EXPANSIONS_PER_TEMPLATE = 28

    samples = []
    cat_counts = Counter()
    for cat_name, (subtype, templates) in CATEGORIES.items():
        seen_for_cat = set()
        for tpl in templates:
            for _ in range(EXPANSIONS_PER_TEMPLATE):
                text = fill(tpl, rng).strip()
                if not text or text in seen_for_cat:
                    continue
                seen_for_cat.add(text)
                samples.append({
                    "payload": text,
                    "subtype": subtype,
                    "category": cat_name,
                    "source": "llm",
                })
        cat_counts[cat_name] = len(seen_for_cat)

    # Cross-category dedup
    seen = set()
    unique = []
    for s in samples:
        if s["payload"] in seen:
            continue
        seen.add(s["payload"])
        s["id"] = "llm_" + hashlib.md5(s["payload"].encode("utf-8")).hexdigest()[:12]
        s["length"] = len(s["payload"])
        unique.append(s)

    # Length filter
    unique = [s for s in unique if 5 <= s["length"] <= 1500]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)

    print(f"Generated LLM-style hard-negative benigns: {len(unique)}")
    print(f"  by category:")
    final_cat = Counter(s["category"] for s in unique)
    for k, n in final_cat.most_common():
        print(f"    {k:20s} {n:>5d}")
    print(f"  total templates: {sum(len(v[1]) for v in CATEGORIES.values())}")
    print(f"  expansions per template: {EXPANSIONS_PER_TEMPLATE}")
    # Length stats
    lens = [s["length"] for s in unique]
    import statistics
    print(f"  length min/median/max: {min(lens)} / {statistics.median(lens):.0f} / {max(lens)}")
    print(f"\n  Wrote {OUT}")


if __name__ == "__main__":
    main()
