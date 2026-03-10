import numpy as np
import pandas as pd
import d_kl_functions as dkl
import os
import avn.similarity as similarity


class Bird:
    def __init__(self, song_dir):
        print(f"[Bird] Initializing Bird with song_dir: {song_dir}")

        self.song_dir = song_dir
        _, self.id = os.path.split(song_dir)

        self.embeddings_dir = os.path.join(self.song_dir, "Embeddings")
        self.segmentation_dir = os.path.join(self.song_dir, "Segmentations")
        self.avn_embedding_file = os.path.join(self.embeddings_dir, dkl.EMBEDDING_FILENAME)

        print(f"[Bird]   id                 = {self.id}")
        print(f"[Bird]   embeddings_dir     = {self.embeddings_dir}")
        print(f"[Bird]   segmentation_dir   = {self.segmentation_dir}")
        print(f"[Bird]   avn_embedding_file = {self.avn_embedding_file}")

        if not os.path.isdir(self.embeddings_dir):
            print(f"[Bird][WARNING] Embeddings directory does not exist: {self.embeddings_dir}")
        if not os.path.isdir(self.segmentation_dir):
            print(f"[Bird][WARNING] Segmentation directory does not exist: {self.segmentation_dir}")

        print()


class Juvenile:
    def __init__(self, juv_dir):
        print(f"[Juvenile] Initializing Juvenile with juv_dir: {juv_dir}")

        self.juv_dir = juv_dir
        _, self.id = os.path.split(juv_dir)
        self.timepoints = []

        for subdir in os.listdir(juv_dir):
            subdir_path = os.path.join(juv_dir, subdir)
            if os.path.isdir(subdir_path):
                self.timepoints.append(Bird(subdir_path))

        print(f"[Juvenile] Added {len(self.timepoints)} timepoints\n")


class Pair:
    def __init__(self, pair_dir):
        print(f"[Pair] Initializing Pair with pair_dir: {pair_dir}")

        _, self.pair_id = os.path.split(pair_dir)
        self.juveniles = []
        self.adults = []
        self.out_dir = pair_dir

        tutor_dir = None
        juv_dir = None

        for subdir in os.listdir(pair_dir):
            subdir_path = os.path.join(pair_dir, subdir)
            subdir_lower = subdir.lower()

            if not os.path.isdir(subdir_path):
                continue

            if "tutor" in subdir_lower:
                tutor_dir = subdir_path
            elif "juv" in subdir_lower or "pupil" in subdir_lower:
                juv_dir = subdir_path
            elif "out" in subdir_lower or "output" in subdir_lower:
                self.out_dir = subdir_path

        if tutor_dir is None:
            raise ValueError(f"No tutor directory found in {pair_dir}")
        if juv_dir is None:
            raise ValueError(f"No juvenile/pupil directory found in {pair_dir}")

        for juv in os.listdir(juv_dir):
            juv_path = os.path.join(juv_dir, juv)
            if os.path.isdir(juv_path):
                self.juveniles.append(Juvenile(juv_path))

        for tutor in os.listdir(tutor_dir):
            tutor_path = os.path.join(tutor_dir, tutor)
            if os.path.isdir(tutor_path):
                self.adults.append(Bird(tutor_path))

        self.kl_df = pd.DataFrame(columns=[
            "Pupil", "Tutor", "D_KL(P|T)", "best_k_P", "best_k_T", "n_P", "n_T"
        ])
        self.df_path = os.path.join(self.out_dir, f"{self.pair_id}.csv")

        print(f"[Pair] Output CSV path: {self.df_path}")
        print(f"[Pair] Juveniles: {len(self.juveniles)}, Tutors: {len(self.adults)}\n")


def precompute_pair_embeddings(pair, model, overwrite=False):
    print("=" * 80)
    print(f"[precompute_pair_embeddings] Pair: {pair.pair_id}")
    print("=" * 80)

    for juv in pair.juveniles:
        for timepoint in juv.timepoints:
            if not os.path.isdir(timepoint.embeddings_dir):
                print(f"[precompute_pair_embeddings][WARNING] Missing embeddings dir: {timepoint.embeddings_dir}")
                continue

            try:
                print(f"[precompute_pair_embeddings] Computing/reusing pupil embeddings for {timepoint.id}")
                dkl.precompute_embeddings_for_bird(
                    bird_id=timepoint.id,
                    embeddings_dir=timepoint.embeddings_dir,
                    model=model,
                    overwrite=overwrite,
                )
            except Exception as e:
                print(f"[precompute_pair_embeddings][ERROR] {timepoint.id}: {type(e).__name__}: {e}")

    for tutor in pair.adults:
        if not os.path.isdir(tutor.embeddings_dir):
            print(f"[precompute_pair_embeddings][WARNING] Missing embeddings dir: {tutor.embeddings_dir}")
            continue

        try:
            print(f"[precompute_pair_embeddings] Computing/reusing tutor embeddings for {tutor.id}")
            dkl.precompute_embeddings_for_bird(
                bird_id=tutor.id,
                embeddings_dir=tutor.embeddings_dir,
                model=model,
                overwrite=overwrite,
            )
        except Exception as e:
            print(f"[precompute_pair_embeddings][ERROR] {tutor.id}: {type(e).__name__}: {e}")


def run_dkl_one_pair(pair_dir, model, overwrite_embeddings=False):
    print("=" * 80)
    print(f"[run_dkl_one_pair] Running on {pair_dir}")
    print("=" * 80)

    pair = Pair(pair_dir)

    # Step 1: compute and save AVN 8D embeddings once
    precompute_pair_embeddings(pair, model=model, overwrite=overwrite_embeddings)

    # Step 2: run DKL on the saved 8D embeddings
    for juv in pair.juveniles:
        for timepoint in juv.timepoints:
            P = timepoint.embeddings_dir

            if not os.path.isdir(P):
                print(f"[run_dkl_one_pair][WARNING] Missing pupil embeddings dir: {P}")
                continue

            for tutor in pair.adults:
                T = tutor.embeddings_dir

                if not os.path.isdir(T):
                    print(f"[run_dkl_one_pair][WARNING] Missing tutor embeddings dir: {T}")
                    continue

                try:
                    print(f"[run_dkl_one_pair] DKL for pupil={timepoint.id}, tutor={tutor.id}")
                    result = dkl.run_dkl(
                        P_dir=P,
                        T_dir=T,
                        P_id=timepoint.id,
                        T_id=tutor.id,
                        model=model,
                        overwrite_embeddings=False,
                        k_range=range(2, 11),
                        n_mc_samples=1000,
                    )

                    pair.kl_df.loc[len(pair.kl_df)] = [
                        timepoint.id,
                        tutor.id,
                        result["dkl_P_given_T"],
                        result["best_k_P"],
                        result["best_k_T"],
                        result["n_P"],
                        result["n_T"],
                    ]

                except Exception as e:
                    print(f"[run_dkl_one_pair][ERROR] pupil={timepoint.id}, tutor={tutor.id}")
                    print(f"[run_dkl_one_pair][ERROR] {type(e).__name__}: {e}")

    print("[run_dkl_one_pair] Final dataframe:")
    print(pair.kl_df)

    pair.kl_df.to_csv(pair.df_path, index=False)
    print(f"[run_dkl_one_pair] Saved: {pair.df_path}\n")


def run_dkl_multiple_pairs(master_dir, overwrite_embeddings=False):
    print("#" * 80)
    print(f"[run_dkl_multiple_pairs] Starting in: {master_dir}")
    print("#" * 80)

    if not os.path.isdir(master_dir):
        raise ValueError(f"Master directory does not exist: {master_dir}")

    print("[run_dkl_multiple_pairs] Loading AVN model once...")
    model = similarity.load_model()
    print("[run_dkl_multiple_pairs] Model loaded.\n")

    for pair_name in os.listdir(master_dir):
        pair_path = os.path.join(master_dir, pair_name)
        if os.path.isdir(pair_path):
            run_dkl_one_pair(pair_path, model=model, overwrite_embeddings=overwrite_embeddings)


if __name__ == "__main__":
    master_dir = r"D:\maria_retutoring\pairs\Done"
    run_dkl_multiple_pairs(master_dir, overwrite_embeddings=False)