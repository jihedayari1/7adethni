#!/usr/bin/env python3
"""
Synthetic Tunisian Derja / Arabizi conversation dataset generator.

Produces Facebook-style threads (post -> comments -> replies) across four topic
types, then derives training-ready conversation pairs.

Everything is SYNTHETIC and flagged needs_native_review=True. These are a
high-quality seed/template meant to be corrected and expanded by native
speakers, not final gold data.
"""

import json

# Generic first names only — no real, identifiable individuals.
THREADS = [
    # ===================== REGULAR TOPICS =====================
    {
        "thread_id": "reg_001",
        "topic_type": "regular",
        "post": {"author": "Yassmine",
                 "text": "el 7ar el yowm ma3adech yetna7mel fi tounes 🥵 chna7welkom enti?"},
        "comments": [
            {"author": "Mehdi",
             "text": "3andek 7a9, hne fi sfax el klim 5addam 24/24, el facture bch ti9tel 😅",
             "replies": [
                 {"author": "Yassmine", "text": "hahaha nafs el 7keya, el steg bch ynaddiwna 3la 7ot"}]},
            {"author": "Ines",
             "text": "rou7ou lel b7ar a7sen, ne7na fel 7ammemet el jaw behi w el nsim ye5i",
             "replies": [
                 {"author": "Mehdi", "text": "weekend ndir niya nahbet lel sahel inchallah"}]},
            {"author": "Karim",
             "text": "ana n9ol el chta a7sen men hedha el 7ar, kahaw 😎", "replies": []},
        ],
    },
    {
        "thread_id": "reg_002",
        "topic_type": "regular",
        "post": {"author": "Sami",
                 "text": "famma chkun y3aref café behi fel marsa na9adou fih el 9ahwa el sob7 ⛅?"},
        "comments": [
            {"author": "Rania",
             "text": "famma wa7ed 9oddem el station, el 9ahwa mte3hom top w el vue 3al b7ar",
             "replies": [
                 {"author": "Sami", "text": "esmou chnowa? bch nroo7 ghodwa njarrbou"}]},
            {"author": "Walid",
             "text": "ana nfaddel el 9hawi el cha3biya, capucin b dinar w noss w barka 😎",
             "replies": [
                 {"author": "Rania", "text": "sa7it ama el ambiance mch kifkif 😂"}]},
        ],
    },
    {
        "thread_id": "reg_003",
        "topic_type": "regular",
        "post": {"author": "Firas",
                 "text": "chkun chef el match el bera7? l3bou behi ama el arbitre 5arrabha 😤"},
        "comments": [
            {"author": "Anis", "text": "el penalty hethika ma kenetch, vol w noss", "replies": []},
            {"author": "Sabri",
             "text": "barka mel zwewel, el équipe mte3na d3ifa w 5alas",
             "replies": [
                 {"author": "Firas", "text": "ma3lich el mohem el niya 😅"}]},
        ],
    },
    {
        "thread_id": "reg_004",
        "topic_type": "regular",
        "post": {"author": "Mariem",
                 "text": "8a3da nra:jel fel révision wel examen ghodwa w ma fhamt walou 🥲 ud3iwli"},
        "comments": [
            {"author": "Nour", "text": "rabbi m3ak 5ti, chwaya 9raya el lila w bch tnajmi", "replies": []},
            {"author": "Hamza",
             "text": "ana nafs el 7ala, el coefficient kbir w el waqt 9sir",
             "replies": [
                 {"author": "Mariem", "text": "courage lina lkol inchallah 💪"}]},
        ],
    },

    # ===================== E-COMMERCE / MARKETPLACE =====================
    {
        "thread_id": "mk_001",
        "topic_type": "marketplace",
        "post": {"author": "Oussama",
                 "text": "👟 nbi3 sabbat Nike original, pointure 42, mosta3mla marrtin barka, b 120 dt. el li mehtem inbox"},
        "comments": [
            {"author": "Bilel",
             "text": "el prix ghali chwaya, na3tik 85 cash",
             "replies": [
                 {"author": "Oussama", "text": "5ouya original moch contrefait, ndir feha 105 w nkemmel m3ak"}]},
            {"author": "Skander",
             "text": "famma livraison l nabeul?",
             "replies": [
                 {"author": "Oussama", "text": "ey famma b 7 dt, wala colis express 3la 7sebek"}]},
            {"author": "Ahmed",
             "text": "el pointure 42 tji kbira wala s8ira?",
             "replies": [
                 {"author": "Oussama", "text": "tji normal, 9is el rjel mte3ek 3adi"}]},
        ],
    },
    {
        "thread_id": "mk_002",
        "topic_type": "marketplace",
        "post": {"author": "Maha",
                 "text": "📱 iPhone 11 64g, état nickel, m3a el chargeur wel boîte, 950 dt négociable chwaya"},
        "comments": [
            {"author": "Ramzi",
             "text": "el batterie 9adech el santé mte3ha?",
             "replies": [
                 {"author": "Maha", "text": "87% mazel behi barcha"}]},
            {"author": "Sofien",
             "text": "850 w nji nesreflek el yowm",
             "replies": [
                 {"author": "Maha", "text": "5alli 900 w hetha e:5er prix 🙏"}]},
        ],
    },
    {
        "thread_id": "mk_003",
        "topic_type": "marketplace",
        "post": {"author": "Hela",
                 "text": "n7eb nechri canapé mosta3mel ama propre, fel grand tunis. el li 3andou ychouf m3aya inbox 🛋️"},
        "comments": [
            {"author": "Nizar",
             "text": "3andi wa7ed 3 places, lon gris, b 300 dt, photos fel inbox", "replies": []},
            {"author": "Yosra",
             "text": "rou7i l souk el jem3a a7sen ti9ay famma 5ir wel as3ar ar5as",
             "replies": [
                 {"author": "Hela", "text": "sa7it ama n5af mel jawda, n7eb wa7ed n7ottou direct"}]},
        ],
    },
    {
        "thread_id": "mk_004",
        "topic_type": "marketplace",
        "post": {"author": "Aymen",
                 "text": "n3mel design l affiches w logos b as3ar mech8oula 🎨 contactini fel inbox lel commande"},
        "comments": [
            {"author": "Dorra",
             "text": "9adech taman logo simple?",
             "replies": [
                 {"author": "Aymen", "text": "yebda men 50 dt 3la 7seb el complexité"}]},
            {"author": "Karim",
             "text": "5dimtek behia, 5demt m3ak el 3am el li fet w kont raji 👌", "replies": []},
        ],
    },

    # ===================== CASUAL STORIES / JOKES =====================
    {
        "thread_id": "sj_001",
        "topic_type": "story_joke",
        "post": {"author": "Ghassen",
                 "text": "el yowm mchit lel 7anout, nsit el portefeuille fel dar, wel 7anouti 3abbeli el couffin kamel 😅 rja3t bel 5fef w ana nest7i"},
        "comments": [
            {"author": "Lina",
             "text": "hahaha 9adha, nafs el 7keya saretli el sim3a el li fet 😂", "replies": []},
            {"author": "Marwen",
             "text": "el 7anouti ze3ma ma 3allamech? 🤣",
             "replies": [
                 {"author": "Ghassen", "text": "3allem ama 3mel rou7ou ma chefnich, rabbi y7afdou 😅"}]},
        ],
    },
    {
        "thread_id": "sj_002",
        "topic_type": "story_joke",
        "post": {"author": "Skander",
                 "text": "nokta: 9alou l jou7a 3lech t7ot el sellom 9oddem el bab? 9alhom bch ne9bet el niveau mte3 el 7ayet 😂😂"},
        "comments": [
            {"author": "Rim", "text": "hahaha 5ayeb ama dh7ekt 🤣", "replies": []},
            {"author": "Tarek",
             "text": "jou7a daymen yji b 7aja 😅",
             "replies": [
                 {"author": "Skander", "text": "el classics ma yfotouch 😎"}]},
        ],
    },
    {
        "thread_id": "sj_003",
        "topic_type": "story_joke",
        "post": {"author": "Asma",
                 "text": "fakartou ki konna s8ar nel3bou 7ata yodlem wel omyet ysi7ou bina mel chra3? 🥹 ayyem el 5ir"},
        "comments": [
            {"author": "Nawfel",
             "text": "wallahi ayyem, el karhba mte3 el plastique wel bteta el cha3biya 😍", "replies": []},
            {"author": "Sirine",
             "text": "el ghbar wel 3ra9 w kont mabsouta akther men tawa 😂",
             "replies": [
                 {"author": "Asma", "text": "sa7it, tawa el sghar 9a3din 3al telephone barka 🙄"}]},
        ],
    },
    {
        "thread_id": "sj_004",
        "topic_type": "story_joke",
        "post": {"author": "Wael",
                 "text": "9olt bch nbda régime el yowm... w sob7 l9it rou9 fel frigo 🍰 el régime ghodwa inchallah 😅"},
        "comments": [
            {"author": "Houda", "text": "ghodwa el li ma yji 3omrou 🤣", "replies": []},
            {"author": "Bechir", "text": "ana 3andi nafs el ghodwa men 3amein 😂", "replies": []},
        ],
    },

    # ===================== QUESTIONS & ANSWERS =====================
    {
        "thread_id": "qa_001",
        "topic_type": "qa",
        "post": {"author": "Nadia",
                 "text": "chkun y3aref tbib asnan mlee7 fi tounes el 3asma w prix m39oul? 3andi waj3a barcha 🦷"},
        "comments": [
            {"author": "Slim",
             "text": "famma Dr fel manar yservi behi wel prix 3adi, n3tik el num fel inbox", "replies": []},
            {"author": "Emna",
             "text": "rou7i l clinique privé a7sen ama 5alli flouss 😅",
             "replies": [
                 {"author": "Nadia", "text": "9adech ya5ou el consultation taghreeban?"},
                 {"author": "Emna", "text": "yebda men 50 l 70 dt 3la 7seb"}]},
        ],
    },
    {
        "thread_id": "qa_002",
        "topic_type": "qa",
        "post": {"author": "Hatem",
                 "text": "el pc mte3i ysi8el barcha w yi79am, chkun 3andou 7all 9bal ma na5dou lel réparateur? 💻"},
        "comments": [
            {"author": "Zied",
             "text": "na77i el programmes el li ti5dem fel démarrage w 3mel nettoyage l disque", "replies": []},
            {"author": "Mona",
             "text": "rubama el RAM 9lila, 9adech 3andek?",
             "replies": [
                 {"author": "Hatem", "text": "4 giga barka 😅"},
                 {"author": "Mona", "text": "hahaha hethi el mochkla, zid RAM w bch yetbaddel el 7al"}]},
        ],
    },
    {
        "thread_id": "qa_003",
        "topic_type": "qa",
        "post": {"author": "Rami",
                 "text": "chkun 3mel passeport jdid el 3am hetha? 9adech ya5ou el délai? 🛂"},
        "comments": [
            {"author": "Sana",
             "text": "ana 5dhitou fi 3 jma3, ama 3la 7seb el baladiya w el za7ma", "replies": []},
            {"author": "Fares",
             "text": "5alli el papiers complets bch ma trawe7ch marrtin, w rou7 sob7 bekri",
             "replies": [
                 {"author": "Rami", "text": "sa7it 3al conseil 🙏"}]},
        ],
    },
    {
        "thread_id": "qa_004",
        "topic_type": "qa",
        "post": {"author": "Leila",
                 "text": "chkun 3andha recette mte3 3ojja behia? n7eb na3melha lel ftour ghodwa 🍳"},
        "comments": [
            {"author": "Olfa",
             "text": "9li el bsal wel felfel, zid el tomatica wel 3dham, w fel e:5er el bidh. top!", "replies": []},
            {"author": "Khaled",
             "text": "zid chwaya harissa bch tji 7ara 😋",
             "replies": [
                 {"author": "Leila", "text": "yaani inchallah n3amelha hakka, merci 🙏"}]},
        ],
    },
]


def normalize_thread(t):
    """Attach IDs and the standard metadata flags to every thread."""
    t["language"] = "tunisian_arabizi"
    t["synthetic"] = True
    t["needs_native_review"] = True
    for ci, c in enumerate(t["comments"], 1):
        c["comment_id"] = f"{t['thread_id']}_c{ci}"
        for ri, r in enumerate(c.get("replies", []), 1):
            r["reply_id"] = f"{c['comment_id']}_r{ri}"
    return t


def derive_pairs(t):
    """Turn a thread into conversation pairs:
       - post  -> each top-level comment
       - comment (with post as context) -> each of its replies
    """
    pairs = []
    post_text = t["post"]["text"]
    for c in t["comments"]:
        pairs.append({
            "topic_type": t["topic_type"],
            "context": "",
            "prompt": post_text,
            "response": c["text"],
            "turn_type": "post->comment",
            "source_thread": t["thread_id"],
        })
        for r in c.get("replies", []):
            pairs.append({
                "topic_type": t["topic_type"],
                "context": f"POST: {post_text}",
                "prompt": c["text"],
                "response": r["text"],
                "turn_type": "comment->reply",
                "source_thread": t["thread_id"],
            })
    return pairs


def main():
    threads = [normalize_thread(t) for t in THREADS]

    # 1) raw threads
    with open("/mnt/user-data/outputs/tunisian_arabizi_threads.jsonl", "w", encoding="utf-8") as f:
        for t in threads:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # 2) derived conversation pairs
    pairs = []
    for i, t in enumerate(threads):
        for j, p in enumerate(derive_pairs(t)):
            p["pair_id"] = f"pair_{i:03d}_{j:02d}"
            p["synthetic"] = True
            p["needs_native_review"] = True
            pairs.append(p)

    with open("/mnt/user-data/outputs/tunisian_arabizi_conversation_pairs.jsonl", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # stats
    from collections import Counter
    by_topic = Counter(t["topic_type"] for t in threads)
    pair_by_topic = Counter(p["topic_type"] for p in pairs)
    print(f"Threads: {len(threads)}  -> {dict(by_topic)}")
    print(f"Pairs:   {len(pairs)}  -> {dict(pair_by_topic)}")
    print("Wrote: tunisian_arabizi_threads.jsonl, tunisian_arabizi_conversation_pairs.jsonl")


if __name__ == "__main__":
    main()
