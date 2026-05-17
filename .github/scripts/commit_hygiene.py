import json
import os
import re
import subprocess
import sys

AI_TERMS = [
    r"claude", r"anthropic", r"\bcodex\b", r"openai", r"chatgpt",
    r"\bgpt[- ]?\d", r"antigravity", r"gemini", r"\bbard\b",
    r"copilot", r"\bcursor\b", r"sourcegraph", r"\bcody\b",
    r"\bdevin\b", r"\baider\b", r"windsurf", r"tabnine",
    r"\bllama\b", r"\bmistral\b", r"\bgrok\b", r"perplexity",
    r"replit", r"ghostwriter", r"\bpieces\b",
]
AI_RE = re.compile("|".join(AI_TERMS), re.IGNORECASE)
TRAILER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9-]*:\s")
MARKER = "<!-- ai-mention-hygiene-check -->"


def find_issues(text):
    issues = []
    for ln in text.splitlines():
        if AI_RE.search(ln):
            issues.append(("ai", ln.strip()))
    lines = text.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    trailers = []
    i = len(lines) - 1
    while i >= 0 and lines[i].strip() != "" and TRAILER_RE.match(lines[i]):
        trailers.append(lines[i])
        i -= 1
    if trailers and i >= 0 and lines[i].strip() == "":
        for t in reversed(trailers):
            issues.append(("trailer", t.strip()))
    return issues


def clean_text(text):
    lines = text.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    trailer_start = len(lines)
    i = len(lines) - 1
    while i >= 0 and lines[i].strip() != "" and TRAILER_RE.match(lines[i]):
        trailer_start = i
        i -= 1
    if trailer_start < len(lines):
        if trailer_start > 0 and lines[trailer_start - 1].strip() == "":
            lines = lines[:trailer_start]
            while lines and not lines[-1].strip():
                lines.pop()
    lines = [l for l in lines if not AI_RE.search(l)]
    out = []
    prev_blank = True
    for l in lines:
        is_blank = (l.strip() == "")
        if is_blank and prev_blank:
            continue
        out.append(l)
        prev_blank = is_blank
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, check=True, **kw)


pr = os.environ["PR_NUMBER"]
repo = os.environ["REPO"]
is_fork = os.environ.get("PR_HEAD_REPO", "") != repo

commits_json = run(
    ["gh", "pr", "view", pr, "--repo", repo, "--json", "commits"]
).stdout
commits = json.loads(commits_json)["commits"]

findings = []
for c in commits:
    sha = c["oid"][:7]
    headline = c.get("messageHeadline") or ""
    body = c.get("messageBody") or ""
    full = headline + (("\n\n" + body) if body else "")
    issues = find_issues(full)
    if issues:
        findings.append({"sha": sha, "subject": headline, "issues": issues})

pr_title = os.environ.get("PR_TITLE", "") or ""
pr_body = os.environ.get("PR_BODY", "") or ""
pr_text = pr_title + (("\n\n" + pr_body) if pr_body else "")
pr_issues = find_issues(pr_text)

sanitized_pr = False
if pr_issues and not is_fork:
    cleaned_title = clean_text(pr_title)
    new_title = cleaned_title if cleaned_title.strip() else pr_title
    new_body = clean_text(pr_body)
    try:
        run(["gh", "pr", "edit", pr, "--repo", repo,
             "--title", new_title, "--body", new_body])
        sanitized_pr = True
        pr_issues = []
        print("::notice::Sanitized PR title and/or body")
    except subprocess.CalledProcessError as e:
        print(f"::warning::Could not auto-sanitize PR title/body: {e.stderr}")

out_lines = [MARKER, "", "## Commit message hygiene check", ""]
clean = not findings and not pr_issues
if clean:
    if sanitized_pr:
        out_lines.append("Auto-cleaned the PR title/body. All commits look fine. :white_check_mark:")
    else:
        out_lines.append("All commit messages and PR text are clean. :white_check_mark:")
else:
    out_lines.append(
        "This PR has commit messages or PR text that violate the repo's"
        " hygiene policy: no trailers (Co-Authored-By, Signed-off-by, any"
        " `--trailer ...`), and no mentions of third-party AI platforms."
    )
    out_lines.append("")
    if findings:
        out_lines.append("### Commits with issues")
        out_lines.append("")
        for f in findings:
            out_lines.append(f"- `{f['sha']}` {f['subject']}")
            for kind, ln in f["issues"]:
                tag = "AI mention" if kind == "ai" else "Trailer"
                out_lines.append(f"  - **{tag}:** `{ln}`")
        out_lines.append("")
    if pr_issues:
        out_lines.append("### PR title / body")
        out_lines.append("")
        for kind, ln in pr_issues:
            tag = "AI mention" if kind == "ai" else "Trailer"
            out_lines.append(f"- **{tag}:** `{ln}`")
        out_lines.append("")
    out_lines.extend([
        "### How to fix",
        "",
        "- For commits: rewrite the branch (e.g. `git rebase -i <base>`),"
        " drop the offending lines from each commit message, then force-push.",
        "- For the PR title/body: just edit the PR description.",
    ])

comment_body = "\n".join(out_lines)

sticky_ids = []
jq_filter = f'.[] | select(.body | contains("{MARKER}")) | .id'
try:
    ids_out = run(["gh", "api", "--paginate",
                   f"repos/{repo}/issues/{pr}/comments",
                   "-q", jq_filter]).stdout
    sticky_ids = [int(x) for x in ids_out.split() if x.strip()]
except subprocess.CalledProcessError as e:
    print(f"::warning::Could not list existing comments: {e.stderr}")

try:
    if sticky_ids:
        for cid in sticky_ids:
            run(["gh", "api", "--method", "PATCH",
                 f"repos/{repo}/issues/comments/{cid}",
                 "-f", f"body={comment_body}"])
    elif not clean:
        run(["gh", "pr", "comment", pr, "--repo", repo,
             "--body", comment_body])
except subprocess.CalledProcessError as e:
    print(f"::warning::Could not post/update comment: {e.stderr}")

if not clean:
    n_commits = len(findings)
    n_pr = len(pr_issues)
    print(f"::error::Found issues in {n_commits} commit(s) and {n_pr} PR text field(s)")
    sys.exit(1)
