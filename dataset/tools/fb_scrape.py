"""
Facebook scraper wrapper (kevinzg/facebook-scraper) -> our pipeline schema.

It pulls POSTS -> COMMENTS -> REPLIES from Tunisian e-commerce pages and writes records
in the SAME shape `clean_facebook.py` already reads, so the two chain together:

    python dataset/tools/fb_scrape.py  --pages pages.txt  --cookies cookies.txt
    python dataset/tools/clean_facebook.py  facebook_raw/*.json

WHAT THIS WRAPPER FIXES (vs raw library use):
  * captures comment REPLIES (the answer side) and flattens comment+reply into ordered rows
  * emits `threadingDepth` (0=comment, 1=reply) + author + likes so pairing works
  * pacing + retry to reduce IP bans and the "content not found" reply errors
  * incremental save (won't lose a 2-hour run if FB blocks mid-way)

WHAT IT CANNOT FIX (Facebook-side, not code):
  * FB serves only a "most relevant" subset of comments anonymously (~30/post) -> scrape MORE posts
  * heavy scraping still risks temporary IP bans -> keep --delay healthy, run in chunks
  * many pages now need login -> pass --cookies (your own FB account; account-ban risk exists)

pages.txt: one page name or post URL per line, e.g.
    mystoretn
    https://www.facebook.com/somestore/posts/123456789
"""
import argparse, json, random, sys, time
from pathlib import Path

import facebook_scraper as fs

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "facebook_raw"          # git-ignored (PII) — never pushed


def comment_likes(c: dict) -> int:
    for k in ("comment_reaction_count", "comment_reactions_count"):
        if c.get(k):
            return int(c[k])
    r = c.get("comment_reactions")
    if isinstance(r, dict):
        return sum(int(v) for v in r.values())
    return 0


def flatten_post(post: dict, page: str) -> list[dict]:
    """One post -> ordered rows: each top-level comment immediately followed by its replies."""
    rows = []
    post_text = (post.get("text") or "").strip()
    for c in post.get("comments_full") or []:
        rows.append({
            "postTitle": post_text, "groupTitle": page,
            "threadingDepth": 0,
            "text": (c.get("comment_text") or "").strip(),
            "profileName": (c.get("commenter_name") or "").strip(),
            "likesCount": comment_likes(c),
            "commentUrl": c.get("comment_url"),
        })
        for r in c.get("replies") or []:
            rows.append({
                "postTitle": post_text, "groupTitle": page,
                "threadingDepth": 1,
                "text": (r.get("comment_text") or "").strip(),
                "profileName": (r.get("commenter_name") or "").strip(),
                "likesCount": comment_likes(r),
                "commentUrl": r.get("comment_url"),
            })
    return rows


def scrape_target(target: str, posts_per_page: int, cookies, delay: float):
    """Yield flattened rows for a page name or a single post URL."""
    opts = {"comments": True, "progress": False, "allow_extra_requests": True}
    is_url = target.startswith("http")
    kwargs = dict(options=opts, cookies=cookies)
    if is_url:
        gen = fs.get_posts(post_urls=[target], **kwargs)
    else:
        gen = fs.get_posts(target, pages=max(1, posts_per_page // 4), **kwargs)

    seen_posts = 0
    while True:
        try:
            post = next(gen)
        except StopIteration:
            break
        except Exception as e:                    # FB hiccup on one post -> skip, keep going
            print(f"    ! post error: {str(e)[:80]}")
            time.sleep(delay * 2)
            continue
        rows = flatten_post(post, target if not is_url else (post.get("username") or target))
        print(f"    post {post.get('post_id','?')}: {len(rows)} comment/reply rows")
        yield rows
        seen_posts += 1
        if not is_url and seen_posts >= posts_per_page:
            break
        time.sleep(delay + random.uniform(0, delay))   # jittered pacing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", required=True, help="file: one page name / post URL per line")
    ap.add_argument("--cookies", default=None, help="cookies.txt (Netscape) or 'from_browser'")
    ap.add_argument("--posts-per-page", type=int, default=30)
    ap.add_argument("--delay", type=float, default=4.0, help="base seconds between posts")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    targets = [l.strip() for l in open(args.pages, encoding="utf-8")
               if l.strip() and not l.startswith("#")]
    OUTDIR.mkdir(exist_ok=True)
    out = Path(args.out) if args.out else OUTDIR / f"batch_{int(time.time())}.json"

    all_rows, total = [], 0
    for i, tgt in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] {tgt}")
        try:
            for rows in scrape_target(tgt, args.posts_per_page, args.cookies, args.delay):
                all_rows += rows
                total += len(rows)
                out.write_text(json.dumps(all_rows, ensure_ascii=False), encoding="utf-8")  # incremental
        except Exception as e:
            print(f"  !! page failed: {str(e)[:100]}")
        print(f"  running total: {total} rows  (saved -> {out.name})")

    depth1 = sum(r["threadingDepth"] == 1 for r in all_rows)
    print(f"\nDONE. {total} rows ({depth1} replies). -> {out}")
    print(f"Next: python dataset/tools/clean_facebook.py {out}")
    if depth1 == 0:
        print("WARNING: 0 replies captured -> no answer side. Pass --cookies and pick posts "
              "that visibly have replies; FB may be withholding them anonymously.")


if __name__ == "__main__":
    main()
