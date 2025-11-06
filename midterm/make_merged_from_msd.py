import sqlite3, argparse
from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/working/merged.csv")

def load_playcounts_from_triplets(path, max_lines=None):
    counts = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if (max_lines is not None) and (i >= max_lines):
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            # user_id = parts[0]
            song_id = parts[1]
            try:
                plays = int(parts[2])
            except ValueError:
                continue
            counts[song_id] = counts.get(song_id, 0) + plays
    return pd.DataFrame(list(counts.items()), columns=["song_id","playcount"])

def load_unique_tracks_mapping(path):
    # Format: track_id <SEP> song_id <SEP> artist_name <SEP> title
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("<SEP>")
            if len(parts) < 2:
                continue
            track_id, song_id = parts[0], parts[1]
            rows.append((song_id, track_id))
    return pd.DataFrame(rows, columns=["song_id","track_id"])

def load_track_metadata(sqlite_path):
    con = sqlite3.connect(sqlite_path)
    cols = ["track_id","duration","year","artist_familiarity","artist_hotttnesss"]
    df = pd.read_sql_query(f"SELECT {', '.join(cols)} FROM songs", con)
    con.close()
    # convert 0 years to NA
    df.loc[df["year"] == 0, "year"] = pd.NA
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triplets", default=RAW / "train_triplets.txt")
    ap.add_argument("--unique",   default=RAW / "unique_tracks.txt")
    ap.add_argument("--meta",     default=RAW / "track_metadata.db")
    ap.add_argument("--max_lines", type=int, default=None)
    args = ap.parse_args()

    print("1) aggregating playcounts from triplets…")
    pc = load_playcounts_from_triplets(str(args.triplets), max_lines=args.max_lines)

    print("2) loading unique_tracks mapping…")
    map_df = load_unique_tracks_mapping(str(args.unique))

    print("3) mapping song_id → track_id and summing per track_id…")
    pc_tid = pc.merge(map_df, on="song_id", how="inner").groupby("track_id", as_index=False)["playcount"].sum()

    print("4) loading track metadata…")
    meta = load_track_metadata(str(args.meta))

    print("5) joining metadata with playcounts…")
    merged = pc_tid.merge(meta, on="track_id", how="inner")

    keep = ["track_id","playcount","duration","year","artist_familiarity","artist_hotttnesss"]
    merged = merged[keep].dropna(subset=["duration","year"])
    merged = merged[(merged["duration"] > 10) & (merged["duration"] < 3600)]
    merged = merged[(merged["year"] >= 1950) & (merged["year"] <= 2025)]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT, index=False)
    print(f"✅ wrote {len(merged):,} rows → {OUT}")

if __name__ == "__main__":
    main()
