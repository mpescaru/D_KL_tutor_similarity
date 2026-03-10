import os
import glob
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# Cached stacked embedding file written inside each Embeddings/ directory
EMBEDDING_FILENAME = "avn_embeddings.npy"


def _is_valid_vector(arr, expected_dim=8):
    """
    Return True if arr can be squeezed into a 1D vector of length expected_dim.
    """
    arr = np.asarray(arr).squeeze()
    return arr.ndim == 1 and arr.shape[0] == expected_dim


def _list_embedding_files(embeddings_dir):
    """
    List per-syllable .npy files in embeddings_dir, excluding the cached combined file.
    """
    files = sorted(glob.glob(os.path.join(embeddings_dir, "*.npy")))
    files = [
        f for f in files
        if os.path.basename(f) != EMBEDDING_FILENAME
    ]
    return files


def load_embedding_dir(embeddings_dir, expected_dim=8, prefer_cached=True):
    """
    Load embeddings from a directory.

    Priority:
      1. cached stacked file: avn_embeddings.npy
      2. stack all valid per-syllable .npy files

    Returns
    -------
    X : ndarray, shape (n_syllables, expected_dim)
    """
    if not os.path.isdir(embeddings_dir):
        raise ValueError(f"Embeddings directory does not exist: {embeddings_dir}")

    cached_path = os.path.join(embeddings_dir, EMBEDDING_FILENAME)

    # 1) Use cached stacked file if available
    if prefer_cached and os.path.isfile(cached_path):
        X = np.load(cached_path)
        X = np.asarray(X)

        if X.ndim != 2 or X.shape[1] != expected_dim:
            raise ValueError(
                f"Cached embedding file has wrong shape: {cached_path}, got {X.shape}, "
                f"expected (n, {expected_dim})"
            )

        if X.shape[0] == 0:
            raise ValueError(f"Cached embedding file is empty: {cached_path}")

        return X.astype(np.float64)

    # 2) Otherwise stack individual files
    files = _list_embedding_files(embeddings_dir)
    if not files:
        raise ValueError(
            f"No per-syllable .npy files found in {embeddings_dir} "
            f"and no cached file {EMBEDDING_FILENAME} exists."
        )

    X = []
    bad_files = []

    for f in files:
        try:
            arr = np.load(f, allow_pickle=False)
            arr = np.asarray(arr).squeeze()

            if arr.ndim == 1 and arr.shape[0] == expected_dim:
                X.append(arr.astype(np.float64))
            else:
                bad_files.append((f, arr.shape))
        except Exception as e:
            bad_files.append((f, f"LOAD_ERROR: {type(e).__name__}: {e}"))

    if len(X) == 0:
        msg = "\n".join([f"  {f} -> {shape}" for f, shape in bad_files[:20]])
        raise ValueError(
            f"No valid {expected_dim}D embeddings found in {embeddings_dir}.\n"
            f"Examples of invalid files:\n{msg}"
        )

    X = np.vstack(X)
    return X


def save_cached_embeddings(embeddings_dir, X):
    """
    Save stacked embeddings to avn_embeddings.npy
    """
    os.makedirs(embeddings_dir, exist_ok=True)
    cached_path = os.path.join(embeddings_dir, EMBEDDING_FILENAME)
    np.save(cached_path, np.asarray(X, dtype=np.float32))
    return cached_path


def precompute_embeddings_for_bird(bird_id, embeddings_dir, model=None, overwrite=False, expected_dim=8):
    """
    Consolidate individual syllable embedding .npy files into one cached stacked file.

    Notes
    -----
    For your current workflow, this function assumes the per-syllable 8D embeddings
    already exist in `embeddings_dir`. The `model` argument is accepted for API
    compatibility with the retutoring script, but is not used here.

    Returns
    -------
    cached_path : str
        Path to saved avn_embeddings.npy
    """
    cached_path = os.path.join(embeddings_dir, EMBEDDING_FILENAME)

    if os.path.isfile(cached_path) and not overwrite:
        print(f"[precompute_embeddings_for_bird] Reusing cached embeddings for {bird_id}: {cached_path}")
        return cached_path

    print(f"[precompute_embeddings_for_bird] Building cached embeddings for {bird_id}")
    X = load_embedding_dir(embeddings_dir, expected_dim=expected_dim, prefer_cached=False)
    cached_path = save_cached_embeddings(embeddings_dir, X)
    print(f"[precompute_embeddings_for_bird] Saved {X.shape[0]} embeddings to {cached_path}")
    return cached_path


def _prepare_joint_standardization(P, T, standardize=True):
    """
    Jointly standardize P and T so they stay in the same coordinate system.
    """
    if not standardize:
        return P, T, None

    scaler = StandardScaler()
    scaler.fit(np.vstack([P, T]))
    Pz = scaler.transform(P)
    Tz = scaler.transform(T)
    return Pz, Tz, scaler


def fit_best_gmm(
    X,
    k_range=range(1, 11),
    random_state=0,
    n_init=5,
    max_iter=1000,
    reg_covar=1e-6
):
    """
    Fit GMMs across k_range and select the one with lowest BIC.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    n_samples = X.shape[0]
    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples to fit a GMM, got {n_samples}")

    candidate_ks = sorted(set(int(k) for k in k_range if int(k) >= 1 and int(k) <= n_samples))
    if not candidate_ks:
        raise ValueError(
            f"No valid k values in k_range={list(k_range)} for n_samples={n_samples}"
        )

    best_gmm = None
    best_bic = np.inf
    bic_table = []

    for k in candidate_ks:
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=random_state,
                n_init=n_init,
                max_iter=max_iter,
                reg_covar=reg_covar,
            )
            gmm.fit(X)
            bic = gmm.bic(X)
            bic_table.append((k, bic))

            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm

        except Exception as e:
            bic_table.append((k, f"ERROR: {type(e).__name__}: {e}"))

    if best_gmm is None:
        raise RuntimeError(f"All GMM fits failed. BIC table: {bic_table}")

    return best_gmm, best_bic, bic_table


def monte_carlo_kl(reference_gmm, comparison_gmm, n_samples=10000):
    """
    Estimate D_KL(reference || comparison) via Monte Carlo:
        E_{x ~ reference} [log p_ref(x) - log p_cmp(x)]
    """
    X_ref, _ = reference_gmm.sample(n_samples=n_samples)
    log_p_ref = reference_gmm.score_samples(X_ref)
    log_p_cmp = comparison_gmm.score_samples(X_ref)
    return float(np.mean(log_p_ref - log_p_cmp))


def run_dkl(
    P_dir,
    T_dir,
    P_id=None,
    T_id=None,
    model=None,
    overwrite_embeddings=False,
    expected_dim=8,
    standardize=True,
    k_range=range(1, 11),
    n_mc_samples=10000,
    random_state=0,
):
    """
    Compute KL divergence between pupil (P) and tutor (T) distributions.

    Returns both directions:
      - dkl_T_given_P = D_KL(T || P)
      - dkl_P_given_T = D_KL(P || T)

    Parameters
    ----------
    P_dir, T_dir : str
        Embeddings directories for pupil and tutor.
    model : optional
        Accepted for compatibility but not used unless you later extend preprocessing.
    overwrite_embeddings : bool
        If True, rebuild cached stacked embedding files.
    """
    if P_id is None:
        P_id = os.path.basename(os.path.normpath(P_dir))
    if T_id is None:
        T_id = os.path.basename(os.path.normpath(T_dir))

    print("=" * 80)
    print(f"[run_dkl] Pupil={P_id}, Tutor={T_id}")
    print(f"[run_dkl] P_dir={P_dir}")
    print(f"[run_dkl] T_dir={T_dir}")

    # Build / reuse cached combined files
    precompute_embeddings_for_bird(P_id, P_dir, model=model, overwrite=overwrite_embeddings, expected_dim=expected_dim)
    precompute_embeddings_for_bird(T_id, T_dir, model=model, overwrite=overwrite_embeddings, expected_dim=expected_dim)

    # Load embeddings
    P = load_embedding_dir(P_dir, expected_dim=expected_dim, prefer_cached=True)
    T = load_embedding_dir(T_dir, expected_dim=expected_dim, prefer_cached=True)

    print(f"[run_dkl] Loaded P shape: {P.shape}")
    print(f"[run_dkl] Loaded T shape: {T.shape}")

    # Joint standardization
    P_use, T_use, _ = _prepare_joint_standardization(P, T, standardize=standardize)

    # Fit GMMs with BIC model selection
    gmm_P, bic_P, bic_table_P = fit_best_gmm(
        P_use,
        k_range=k_range,
        random_state=random_state
    )

    gmm_T, bic_T, bic_table_T = fit_best_gmm(
        T_use,
        k_range=k_range,
        random_state=random_state
    )

    print(f"[run_dkl] Best k_P = {gmm_P.n_components}, BIC = {bic_P}")
    print(f"[run_dkl] Best k_T = {gmm_T.n_components}, BIC = {bic_T}")

    # KL estimates in both directions
    dkl_T_given_P = monte_carlo_kl(gmm_T, gmm_P, n_samples=n_mc_samples)
    dkl_P_given_T = monte_carlo_kl(gmm_P, gmm_T, n_samples=n_mc_samples)

    print(f"[run_dkl] D_KL(T || P) = {dkl_T_given_P}")
    print(f"[run_dkl] D_KL(P || T) = {dkl_P_given_T}")

    return {
        "Pupil": P_id,
        "Tutor": T_id,
        "dkl_T_given_P": float(dkl_T_given_P),
        "dkl_P_given_T": float(dkl_P_given_T),
        "best_k_P": int(gmm_P.n_components),
        "best_k_T": int(gmm_T.n_components),
        "bic_P": float(bic_P),
        "bic_T": float(bic_T),
        "bic_table_P": bic_table_P,
        "bic_table_T": bic_table_T,
        "n_P": int(P.shape[0]),
        "n_T": int(T.shape[0]),
        "standardized": bool(standardize),
        "embedding_dim": int(expected_dim),
        "n_mc_samples": int(n_mc_samples),
    }
