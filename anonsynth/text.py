"""Short synthetic posts with planted marker tokens and one planted spurious shortcut."""
from __future__ import annotations

import numpy as np
import pandas as pd

NEUTRAL = (
    "the a to and of in is it for on with this that just my was so but not have are be at "
    "all from about what when out up new time people day today really good one more get "
    "like see back think know want need still way going make look first last great never"
).split()
BOT_MARKERS = ["giveaway", "promo", "crypto", "click", "winner", "dm"]
TROLL_MARKERS = ["idiot", "sheep", "liar", "shill", "clown", "wake"]
SPURIOUS_TOKEN = "#trending"

KIND_HUMAN, KIND_BOT, KIND_TROLL = 0, 1, 2


def make_posts(account_ids, kind, split, cfg, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    pid = 0
    for aid, k, sp in zip(account_ids, kind, split):
        inauth = k != KIND_HUMAN
        if k == KIND_BOT:
            markers, p_marker = BOT_MARKERS, cfg.marker_prob_inauthentic
        elif k == KIND_TROLL:
            markers, p_marker = TROLL_MARKERS, cfg.marker_prob_inauthentic
        else:
            markers, p_marker = BOT_MARKERS + TROLL_MARKERS, cfg.marker_prob_background
        if inauth:
            p_spur = (
                cfg.spurious_prob_test_inauth
                if sp == "test"
                else cfg.spurious_prob_train_inauth
            )
        else:
            p_spur = cfg.spurious_prob_authentic
        for _ in range(cfg.posts_per_account):
            tokens = list(rng.choice(NEUTRAL, size=int(rng.integers(6, 15))))
            planted = []
            if rng.random() < p_marker:
                m = str(rng.choice(markers))
                tokens.insert(int(rng.integers(0, len(tokens) + 1)), m)
                planted.append(m)
            has_spur = bool(rng.random() < p_spur)
            if has_spur:
                tokens.insert(int(rng.integers(0, len(tokens) + 1)), SPURIOUS_TOKEN)
            rows.append((pid, aid, " ".join(tokens), "|".join(planted), has_spur))
            pid += 1
    return pd.DataFrame(
        rows, columns=["post_id", "account_id", "text", "planted_markers", "has_spurious_token"]
    )
