import os
import pandas as pd
import d_kl_functions as dkl
import avn.similarity as similarity


class Bird:
    """
    Represents one song directory (either a tutor or one juvenile timepoint).
    Expected structure:
        song_dir/
            Embeddings/
                1.npy
                2.npy
                ...
            Segmentations/   # optional for current workflow
    """
    def __init__(self, song_dir):
        self.song_dir = song_dir
        _, self.id = os.path.split(os.path.normpath(song_dir))

        self.embeddings_dir = os.path.join(self.song_dir, "Embeddings")
        self.segmentation_dir = os.path.join(self.song_dir, "Segmentations")
        self.avn_embedding_file = os.path.join(self.embeddings_dir, dkl.EMBEDDING_FILENAME)

        print(f"[Bird] {self.id}")
        print(f"       song_dir          = {self.song_dir}")
        print(f"       embeddings_dir    = {self.embeddings_dir}")
        print(f"       segmentation_dir  = {self.segmentation_dir}")
        print(f"       cached_embeddings = {self.avn_embedding_file}")

        if not os.path.isdir(self.embeddings_dir):
            print(f"[Bird][WARNING] Missing embeddings directory: {self.embeddings_dir}")


class Juvenile:
    """
    Represents one juvenile bird containing multiple timepoints.
    Expected structure:
        juvenile_dir/
            day1/
            day2/
            day3/
            ...
    """
    def __init__(self, juv_dir):
        self.juv_dir = juv_dir
        _, self.id = os.path.split(os.path.normpath(juv_dir))
        self.timepoints = []

        print(f"[Juvenile] {self.id}")

        for subdir in sorted(os.listdir(juv_dir)):
            subdir_path = os.path.join(juv_dir, subdir)
            if os.path.isdir(subdir_path):
                self.timepoints.append(Bird(subdir_path))

        print(f"[Juvenile] Added {len(self.timepoints)} timepoints for {self.id}\n")


class Pair:
    """
    Represents one retutoring pair directory.

    Expected rough structure:
        pair_dir/
            tutor/ or tutors/
                tutorA/
                tutorB/
            juv/ or pupil/
                juvenile1/
                    timepoint1/
                    timepoint2/
                juvenile2/
                    ...
            out/ or output/   # optional
    """
    def __init__(self, pair_dir):
        self.pair_dir = pair_dir
        _, self.pair_id = os.path.split(os.path.normpath(pair_dir))

        self.juveniles = []
        self.adults = []
        self.out_dir = pair_dir  # default if no explicit output folder

        tutor_dir = None
        juv_dir = None

        print("=" * 80)
        print(f"[Pair] Initializing pair: {self.pair_id}")
        print(f"[Pair] pair_dir = {self.pair_dir}")

        for subdir in sorted(os.listdir(pair_dir)):
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

        for juv_name in sorted(os.listdir(juv_dir)):
            juv_path = os.path.join(juv_dir, juv_name)
            if os.path.isdir(juv_path):
                self.juveniles.append(Juvenile(juv_path))

        for tutor_name in sorted(os.listdir(tutor_dir)):
            tutor_path = os.path.join(tutor_dir, tutor_name)
            if os.path.isdir(tutor_path):
                self.adults.append(Bird(tutor_path))

        self.kl_df = pd.DataFrame(columns=[
            "Pair",
            "Juvenile",
            "PupilTimepoint",
            "Tutor",
            "D_KL(T|P)",
            "D_KL(P|T)",
            "best_k_P",
            "best_k_T",
            "bic_P",
            "bic_T",
            "n_P",
            "n_T",
            "standardized",
            "embedding_dim",
            "n_mc_samples",
        ])

        os.makedirs(self.out_dir, exist_ok=True)
        self.df_path = os.path.join(self.out_dir, f"{self.pair_id}.csv")

        print(f"[Pair] tutor_dir = {tutor_dir}")
        print(f"[Pair] juv_dir   = {juv_dir}")
        print(f"[Pair] out_dir   = {self.out_dir}")
        print(f"[Pair] csv_path  = {self.df_path}")
        print(f"[Pair] n_juveniles = {len(self.juveniles)}")
        print(f"[Pair] n_tutors    = {len(self.adults)}")
        print("=" * 80 + "\n")


def precompute_pair_embeddings(pair, model=None, overwrite=False):
    """
    Build/reuse cached avn_embeddings.npy for every tutor and juvenile timepoint.
    """
    print("=" * 80)
    print(f"[precompute_pair_embeddings] Pair: {pair.pair_id}")
    print("=" * 80)

    for juv in pair.juveniles:
        for timepoint in juv.timepoints:
            if not os.path.isdir(timepoint.embeddings_dir):
                print(f"[precompute_pair_embeddings][WARNING] Missing embeddings dir: {timepoint.embeddings_dir}")
                continue

            try:
                dkl.precompute_embeddings_for_bird(
                    bird_id=timepoint.id,
                    embeddings_dir=timepoint.embeddings_dir,
                    model=model,
                    overwrite=overwrite,
                )
            except Exception as e:
                print(f"[precompute_pair_embeddings][ERROR] pupil={timepoint.id} -> {type(e).__name__}: {e}")

    for tutor in pair.adults:
        if not os.path.isdir(tutor.embeddings_dir):
            print(f"[precompute_pair_embeddings][WARNING] Missing embeddings dir: {tutor.embeddings_dir}")
            continue

        try:
            dkl.precompute_embeddings_for_bird(
                bird_id=tutor.id,
                embeddings_dir=tutor.embeddings_dir,
                model=model,
                overwrite=overwrite,
            )
        except Exception as e:
            print(f"[precompute_pair_embeddings][ERROR] tutor={tutor.id} -> {type(e).__name__}: {e}")

    print()


def _append_result_row(df, pair_id, juvenile_id, pupil_timepoint_id, result):
    """
    Append one KL result row to a dataframe.
    """
    row = {
        "Pair": pair_id,
        "Juvenile": juvenile_id,
        "PupilTimepoint": pupil_timepoint_id,
        "Tutor": result["Tutor"],
        "D_KL(T|P)": result["dkl_T_given_P"],
        "D_KL(P|T)": result["dkl_P_given_T"],
        "best_k_P": result["best_k_P"],
        "best_k_T": result["best_k_T"],
        "bic_P": result["bic_P"],
        "bic_T": result["bic_T"],
        "n_P": result["n_P"],
        "n_T": result["n_T"],
        "standardized": result["standardized"],
        "embedding_dim": result["embedding_dim"],
        "n_mc_samples": result["n_mc_samples"],
    }

    df.loc[len(df)] = row
    return df


def run_dkl_one_pair(
    pair_dir,
    model,
    overwrite_embeddings=False,
    k_range=range(1, 11),
    n_mc_samples=10000,
    standardize=True,
    save_after_each_row=True,
):
    """
    Run DKL for all pupil timepoints x all tutors in a single pair directory.
    """
    print("=" * 80)
    print(f"[run_dkl_one_pair] Starting: {pair_dir}")
    print("=" * 80)

    pair = Pair(pair_dir)

    # Cache combined embeddings once per bird/timepoint
    precompute_pair_embeddings(pair, model=model, overwrite=overwrite_embeddings)

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
                    print("-" * 80)
                    print(f"[run_dkl_one_pair] juvenile={juv.id}, timepoint={timepoint.id}, tutor={tutor.id}")

                    result = dkl.run_dkl(
                        P_dir=P,
                        T_dir=T,
                        P_id=timepoint.id,
                        T_id=tutor.id,
                        model=model,
                        overwrite_embeddings=False,
                        k_range=k_range,
                        n_mc_samples=n_mc_samples,
                        standardize=standardize,
                    )

                    pair.kl_df = _append_result_row(
                        pair.kl_df,
                        pair_id=pair.pair_id,
                        juvenile_id=juv.id,
                        pupil_timepoint_id=timepoint.id,
                        result=result,
                    )

                    if save_after_each_row:
                        pair.kl_df.to_csv(pair.df_path, index=False)
                        print(f"[run_dkl_one_pair] Intermediate save -> {pair.df_path}")

                except Exception as e:
                    print(f"[run_dkl_one_pair][ERROR] juvenile={juv.id}, timepoint={timepoint.id}, tutor={tutor.id}")
                    print(f"[run_dkl_one_pair][ERROR] {type(e).__name__}: {e}")

    print("\n[run_dkl_one_pair] Final dataframe:")
    print(pair.kl_df)

    pair.kl_df.to_csv(pair.df_path, index=False)
    print(f"[run_dkl_one_pair] Final save -> {pair.df_path}\n")

    return pair.kl_df


def run_dkl_multiple_pairs(
    master_dir,
    overwrite_embeddings=False,
    k_range=range(1, 11),
    n_mc_samples=10000,
    standardize=True,
):
    """
    Run DKL on all pair directories inside master_dir.
    Loads AVN model once for compatibility / future extension.
    """
    print("#" * 80)
    print(f"[run_dkl_multiple_pairs] master_dir = {master_dir}")
    print("#" * 80)

    if not os.path.isdir(master_dir):
        raise ValueError(f"Master directory does not exist: {master_dir}")

    print("[run_dkl_multiple_pairs] Loading AVN model once...")
    model = similarity.load_model()
    print("[run_dkl_multiple_pairs] Model loaded.\n")

    for pair_name in sorted(os.listdir(master_dir)):
        pair_path = os.path.join(master_dir, pair_name)
        if os.path.isdir(pair_path):
            try:
                run_dkl_one_pair(
                    pair_dir=pair_path,
                    model=model,
                    overwrite_embeddings=overwrite_embeddings,
                    k_range=k_range,
                    n_mc_samples=n_mc_samples,
                    standardize=standardize,
                    save_after_each_row=True,
                )
            except Exception as e:
                print(f"[run_dkl_multiple_pairs][ERROR] pair={pair_name}")
                print(f"[run_dkl_multiple_pairs][ERROR] {type(e).__name__}: {e}")


if __name__ == "__main__":
    master_dir = r"G:\retutoring\pairs"

    run_dkl_multiple_pairs(
        master_dir=master_dir,
        overwrite_embeddings=False,
        k_range=range(1, 11),
        n_mc_samples=1000,
        standardize=True,
    )
