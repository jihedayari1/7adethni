"""
YouTube comment-thread scraper -> raw threads for build_real_conv.py.

Why YouTube: comments on Tunisian creators are REAL, conversational, and usually written
ALREADY in Arabizi (naturally vowelled) — far better generation targets than Arabic-script
text transliterated by code. It uses the official Data API v3 (legal, free quota), not scraping.

Setup
  1. console.cloud.google.com -> new project -> enable "YouTube Data API v3" -> create API key
  2. export YOUTUBE_API_KEY=...        (Windows PowerShell: $env:YOUTUBE_API_KEY="...")

Usage
  python dataset/tools/yt_comments.py VIDEO_ID [VIDEO_ID ...]          # specific videos
  python dataset/tools/yt_comments.py --channel UCxxxx --max 20        # a channel's recent videos
  python dataset/tools/yt_comments.py --channel @SomeHandle --max 20   # handle is resolved for you
  python dataset/tools/yt_comments.py --search "tunisie vlog" --max 30 # search videos by query

Out: aigenerateddataset/yt_threads.jsonl   (then: python dataset/tools/build_real_conv.py aigenerateddataset/yt_threads.jsonl)
"""
import os, sys, json, time
from pathlib import Path
import urllib.request, urllib.parse

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "aigenerateddataset" / "yt_threads.jsonl"
API = "https://www.googleapis.com/youtube/v3"
KEY = os.environ.get("YOUTUBE_API_KEY", "")


def _get(endpoint, **params):
    params["key"] = KEY
    url = f"{API}/{endpoint}?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def video_title(vid):
    try:
        items = _get("videos", part="snippet", id=vid).get("items", [])
        return items[0]["snippet"]["title"] if items else vid
    except Exception:
        return vid


def resolve_channel(token: str) -> str:
    """Accept a UC… id, an @handle, or a plain name -> return the channel id."""
    if token.startswith("UC") and len(token) >= 20:
        return token
    handle = token if token.startswith("@") else "@" + token
    try:
        items = _get("channels", part="id", forHandle=handle).get("items", [])
        if items:
            return items[0]["id"]
    except Exception:
        pass
    # fall back to search
    res = _get("search", part="snippet", q=token.lstrip("@"), type="channel", maxResults=1)
    items = res.get("items", [])
    if not items:
        raise SystemExit(f"could not resolve channel '{token}'")
    return items[0]["snippet"]["channelId"]


def search_videos(query, max_videos):
    """Video ids matching a query (relevance-ranked)."""
    vids, token = [], None
    while len(vids) < max_videos:
        page = _get("search", part="id", q=query, type="video", relevanceLanguage="ar",
                    maxResults=min(50, max_videos), pageToken=token or "")
        vids += [i["id"]["videoId"] for i in page.get("items", []) if i["id"].get("videoId")]
        token = page.get("nextPageToken")
        if not token:
            break
    return vids[:max_videos]


def channel_videos(channel_id, max_videos):
    """Most-recent uploads of a channel (via the uploads playlist)."""
    channel_id = resolve_channel(channel_id)
    ch = _get("channels", part="contentDetails", id=channel_id)["items"][0]
    uploads = ch["contentDetails"]["relatedPlaylists"]["uploads"]
    vids, token = [], None
    while len(vids) < max_videos:
        page = _get("playlistItems", part="contentDetails", playlistId=uploads,
                    maxResults=50, pageToken=token or "")
        vids += [i["contentDetails"]["videoId"] for i in page.get("items", [])]
        token = page.get("nextPageToken")
        if not token:
            break
    return vids[:max_videos]


def fetch_threads(vid, cap=400):
    """Return one thread record: {post, comments:[{text, replies:[{text}]}]}.
    Stops after ~`cap` top-level comments (keeps a diversified pull fast + quota-light)."""
    title = video_title(vid)
    comments, token = [], None
    while len(comments) < cap:
        try:
            page = _get("commentThreads", part="snippet,replies", videoId=vid,
                        maxResults=100, textFormat="plainText", order="relevance",
                        pageToken=token or "")
        except Exception as e:
            print(f"  ! {vid}: {e}")
            break
        for it in page.get("items", []):
            top = it["snippet"]["topLevelComment"]["snippet"]["textOriginal"]
            reps = [r["snippet"]["textOriginal"]
                    for r in (it.get("replies", {}).get("comments", []))]
            comments.append({"text": top, "replies": [{"text": x} for x in reps]})
        token = page.get("nextPageToken")
        if not token:
            break
        time.sleep(0.2)
    return {"post": title, "video_id": vid, "comments": comments}


def main():
    if not KEY:
        print("Set YOUTUBE_API_KEY first (see header).")
        return
    args = sys.argv[1:]
    mx  = int(args[args.index("--max") + 1]) if "--max" in args else 20
    cap = int(args[args.index("--cap") + 1]) if "--cap" in args else 400
    if "--channel" in args:
        vids = channel_videos(args[args.index("--channel") + 1], mx)
    elif "--search" in args:
        vids = search_videos(args[args.index("--search") + 1], mx)
    else:
        vids = [a for a in args if not a.startswith("--")]
    if not vids:
        print("Pass VIDEO_IDs, --channel <id|@handle>, or --search \"query\"."); return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_threads = n_comments = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for vid in vids:
            t = fetch_threads(vid, cap=cap)
            n_threads += 1
            n_comments += sum(1 + len(c["replies"]) for c in t["comments"])
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
            print(f"  {vid}: {len(t['comments'])} top-level comments")
    print(f"\n{n_threads} videos -> {n_comments} comments -> {OUT.relative_to(ROOT)}")
    print("Next: python dataset/tools/build_real_conv.py", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
