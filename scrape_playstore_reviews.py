"""
Play Store Game Reviews Scraper
================================
Collects 1000+ reviews from Google Play Store games and saves them to:
    reviews_dataset.csv

Labeling rule (rating-based):
    1-2  → negative
    3    → neutral
    4-5  → positive

Requirements:
    pip install google-play-scraper pandas

Usage:
    python scrape_playstore_reviews.py
"""

import csv
import time
import random
import pandas as pd
from google_play_scraper import reviews, Sort

# ── Games to scrape (app_id, display_name, category) ──────────────────────
GAMES = [
    ("com.supercell.clashofclans",      "Clash of Clans",      "Strategy"),
    ("com.king.candycrushsaga",         "Candy Crush Saga",    "Puzzle"),
    ("com.tencent.ig",                  "PUBG Mobile",         "Battle Royale"),
    ("com.kiloo.subwaysurf",            "Subway Surfers",      "Endless Runner"),
    ("com.innersloth.spacemafia",       "Among Us",            "Social Deduction"),
    ("com.roblox.client",               "Roblox",              "Sandbox"),
    ("com.mojang.minecraftpe",          "Minecraft",           "Sandbox"),
    ("com.miHoYo.GenshinImpact",        "Genshin Impact",      "Action RPG"),
    ("com.activision.callofduty.shooter","Call of Duty Mobile","Shooter"),
    ("com.nianticlabs.pokemongo",       "Pokémon GO",          "Augmented Reality"),
    ("com.gameloft.android.ANMP.GloftA9HM", "Asphalt 9",      "Racing"),
    ("com.dts.freefireth",              "Free Fire",           "Battle Royale"),
    ("com.supercell.brawlstars",        "Brawl Stars",         "Action"),
    ("com.supercell.clashroyale",       "Clash Royale",        "Strategy"),
    ("com.mobile.legends",              "Mobile Legends",      "MOBA"),
]

# ── Target counts per label to keep classes balanced ──────────────────────
# We aim for ~350 per label = ~1050 total
# Over-collect then trim to balance.
REVIEWS_PER_APP = 350          # fetched per app per call
MAX_TOTAL       = 3000         # safety cap before balancing

def assign_label(rating: int) -> str:
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:                       # 4 or 5
        return "positive"


def scrape_all_games() -> list[dict]:
    all_rows = []
    review_counter = 1

    for app_id, game_name, category in GAMES:
        print(f"  Scraping: {game_name} ...", end=" ", flush=True)
        try:
            result, _ = reviews(
                app_id,
                lang="en",
                country="us",
                sort=Sort.MOST_RELEVANT,
                count=REVIEWS_PER_APP,
                filter_score_with=None,   # all ratings
            )
            fetched = 0
            for r in result:
                text   = (r.get("content") or "").strip()
                rating = r.get("score")

                if not text or not rating:
                    continue

                all_rows.append({
                    "review_id":        f"REV{review_counter:05d}",
                    "source":           "Google Play Store",
                    "product_category": f"Games – {category}",
                    "review_text":      text,
                    "rating":           rating,
                    "label":            assign_label(rating),
                })
                review_counter += 1
                fetched += 1

            print(f"{fetched} reviews collected.")
        except Exception as e:
            print(f"FAILED ({e})")

        # Be polite — avoid hammering the endpoint
        time.sleep(random.uniform(1.5, 3.0))

        if len(all_rows) >= MAX_TOTAL:
            print("  Reached safety cap. Stopping early.")
            break

    return all_rows


def balance_classes(rows: list[dict], target_per_class: int = 370) -> list[dict]:
    """Trim each label class to target_per_class rows (or fewer if not enough)."""
    df = pd.DataFrame(rows)
    balanced = (
        df.groupby("label", group_keys=False)
          .apply(lambda g: g.sample(min(len(g), target_per_class), random_state=42))
    )
    # Re-assign sequential IDs after shuffling
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    balanced["review_id"] = [f"REV{i+1:05d}" for i in range(len(balanced))]
    return balanced.to_dict(orient="records")


def save_csv(rows: list[dict], path: str = "reviews_dataset.csv") -> None:
    fieldnames = ["review_id", "source", "product_category",
                  "review_text", "rating", "label"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✅  Saved {len(rows)} reviews → {path}")


def print_summary(rows: list[dict]) -> None:
    from collections import Counter
    labels  = Counter(r["label"]  for r in rows)
    ratings = Counter(r["rating"] for r in rows)
    print("\n── Dataset Summary ─────────────────────────────")
    print(f"  Total reviews  : {len(rows)}")
    print(f"  Label dist     : {dict(labels)}")
    print(f"  Rating dist    : {dict(sorted(ratings.items()))}")
    print("────────────────────────────────────────────────")


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🎮  Play Store Reviews Scraper")
    print("=" * 40)
    print(f"  Scraping {len(GAMES)} games …\n")

    raw_rows = scrape_all_games()

    print(f"\n  Raw total collected : {len(raw_rows)}")
    print("  Balancing classes …")

    balanced_rows = balance_classes(raw_rows, target_per_class=370)

    print_summary(balanced_rows)
    save_csv(balanced_rows, "reviews_dataset.csv")
