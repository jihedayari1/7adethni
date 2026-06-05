# Scraping spec — Tunisian customer-service assistant

**Product goal:** an LLM that understands + speaks Tunisian (Derja/Arabizi) for
**e-commerce customer service** (answer clients automatically), sellable to Tunisian
stores, and later a public subscription assistant.

**What teaches it that:** real **customer question → business answer** pairs in Derja.
That is exactly what Facebook e-commerce pages contain. So scraping fits this goal.

---

## 1. WHERE to scrape (sources)
Active **Tunisian business / e-commerce Facebook pages** whose comment sections are full of
customer Q&A. Spread across **30–50 pages and many categories** (diversity matters more than
depth on one page):

- Clothing / fashion shops (women, men, kids)
- Phones / electronics / accessories
- Cosmetics / parfums / skincare
- Food delivery / restaurants / pâtisserie
- Baby & mom products
- Home / furniture / deco
- Sport / supplements
- Shoes / bags
- Marketplace & buy/sell groups (e.g. general "vente Tunisie" groups)

Pick pages where the **page itself replies to comments** (you'll see the shop answering
"prix?", "disponible?", "livraison?"). Those replies are your gold answers.

> Privacy rule (non-negotiable): collect **public** posts/comments only. We strip ALL personal
> data before anything is used or stored long-term. Raw scrapes never go to a public repo.

---

## 2. WHAT to capture per item (raw schema)
Scrape **comments WITH their reply threads** (not flat) so we can link question→answer.
Keep these fields per comment:

| field | why |
|-------|-----|
| `post_id` | group comments by post |
| `post_text` | the post (often the product) = context for the question |
| `page_name` | to detect which replies are the SHOP's answer |
| `comment_id` | unique id |
| `parent_comment_id` | links a reply to the question it answers ⭐ |
| `thread_depth` | 0 = top-level (usually the question), 1+ = reply |
| `text` | the actual comment ⭐ |
| `author_name` | **temporary** — only to detect "is this the shop replying?", then DELETED |
| `likes_count` | rank best answers |
| `date` | recency / dedup |
| `source_url` | provenance |

Your existing Apify "facebook-comments-scraper" already returns most of these
(`postTitle`, `text`, `threadingDepth`, `profileName`, `likesCount`…). Make sure replies
are included and that `parent`/threading is captured.

---

## 3. HOW a pair is formed (pairing logic)
```
QUESTION = a top-level comment (thread_depth 0) that asks something
           ("9adech?", "disponible?", "chnowa el mqas?", "livraison l Sfax?")
ANSWER   = a reply to that comment, preferring:
             1) a reply written by the PAGE itself (author_name == page_name)
             2) else the highest-liked relevant reply
CONTEXT  = post_text (optional, helps when the question is about the product)
```
→ becomes one training row (PII stripped):
```json
{ "context": "<post_text>", "question": "<customer comment>",
  "answer": "<shop reply>", "category": "price|delivery|availability|...",
  "likes": 4, "source_page": "<page>" }
```

---

## 4. HOW MANY rows (the number you asked for)
Facebook comments are noisy — after filtering, only **~15–20%** become usable pairs.
So you must scrape **~5–6× your clean target.**

| Stage | Clean pairs (after filtering) | Raw comments to scrape | Result |
|-------|------------------------------|-------------------------|--------|
| **Alpha** (prove it talks) | 3,000 – 5,000 | ~25,000 – 40,000 | clearly speaks Tunisian CS, rough |
| **Sellable v1** ⭐ | **10,000 – 20,000** | **~70,000 – 120,000** | reliable for real store chats |
| **Strong product** | 30,000 – 50,000 | ~200,000+ | robust, handles edge cases |

Aim for **Sellable v1 first: ~80,000–120,000 raw comments** across 30–50 pages.
Claude then **augments** the clean seed (rephrases, fills rare intents) to multiply it ~2–3×,
so you don't need to scrape the full 50k for a strong model.

**You only handwrite ~200–300 pairs total — for the EVALUATION set (testing), not training.**

---

## 5. Coverage checklist (make sure the scrape spans these intents)
A store bot must handle all of these. Track how many pairs you have per intent:

`greeting` · `product availability` · `price` · `sizes / colors / variants` ·
`product details / specs` · `delivery time` · `delivery cost` · `delivery zones` ·
`payment methods (D17, flouci, cash on delivery…)` · `place an order` ·
`order status / tracking` · `modify / cancel order` · `returns / refunds` · `warranty` ·
`complaints` · `store hours / location` · `promotions / discounts` · `thanks / closing`

If some intents are thin after scraping, that's where Claude generation fills the gap.

---

## 6. What we DROP in filtering (so you know the yield)
- answers under ~2 words, or only emoji / "@name" tag-a-friend
- spam, links-only, insults, fights
- duplicates / near-duplicates
- pairs where the reply doesn't actually answer the question
- pure French/English/MSA-only (we KEEP Derja with French code-switching — that's normal)

---

## 7. Where the model gets FACTS right (RAG — already built)
Fine-tuning teaches it **to talk like a Tunisian shop**. It must NOT invent prices/policies.
Each store's real **catalog + prices + delivery + policies** go into the RAG layer
(the retriever we already built generalizes from "lexicon" to "product knowledge"), and get
injected per-answer. So: **fine-tune = Tunisian style; RAG = correct facts per store.**

---

## 8. Scraper settings — fix what the pilot exposed
A test run on the first 667-comment scrape produced **0 pairs**. Diagnosis:
- **all comments were `threadingDepth: 0`** → the scraper captured **no replies**, so the
  *answer* side was entirely missing. **This is the must-fix.**
- only **3 posts**, from a **telecom complaints** page (wrong domain).

When you configure the Facebook comments scraper (e.g. Apify), make sure to:
1. ✅ **Enable replies / nested comments** ("Include comment replies" / `includeNestedComments`).
   Without this you only get questions, never answers → 0 pairs.
2. ✅ Set a **high max comments + max replies per post** (e.g. 200+ / 50+).
3. ✅ Feed **many post URLs** from **30–50 e-commerce pages** (see §1), not 3 posts.
4. ✅ Keep **author name + reply/parent info** in the output (so pairing + shop-reply detection
   work). I delete the names during cleaning.
5. ✅ Pick posts where you can SEE the shop answering comments (prix/dispo/livraison).

Then run: `python dataset/tools/clean_facebook.py <your_new_scrape.json>` — it strips PII and
builds the pairs automatically.

### Two scraper options
- **Apify "facebook-comments-scraper" (paid, robust)** — what you already used; just enable
  replies. More reliable against FB blocking; best for hitting volume for a paid product.
- **kevinzg/facebook-scraper (free) — CONFIRMED NON-VIABLE (tested 2026-06).**
  Cloned + live-tested: `get_posts()` returns 0 posts. Facebook now serves a **login wall** for
  both `m.facebook.com` and `mbasic.facebook.com`, so its "scrape without an API key" premise is
  dead. Repo is also unmaintained (last commit Oct 2023) with stale HTML selectors. Not worth
  repairing — would need live login cookies (account-ban risk) + a full extractor rewrite that
  breaks on FB's next change. The `dataset/tools/fb_scrape.py` wrapper remains only as a schema
  adapter if a working scraper is swapped in.
- **Official Meta Graph/Messenger API — the production path.** For the shipped product, each store
  authorizes access to its OWN page/Messenger via Meta's official API: legal, reliable, and it's
  the integration the bot needs to read/reply to messages anyway. Scraping is only a one-time
  training-data bootstrap; the product runs on the official API.

## TL;DR — what to bring me
1. **~80k–120k raw FB comments** (with reply threads) from 30–50 Tunisian e-commerce pages,
   covering the intents in §5. Keep `author_name` + `parent` so I can build pairs; I strip PII.
2. That's it for training data. I handle cleaning, pairing, filtering, and Claude augmentation.
3. Separately, ~200–300 handwritten gold pairs later, only for testing.
