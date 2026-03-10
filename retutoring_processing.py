import numpy as np
import pandas as pd
import d_kl_functions as dkl
import os

class Bird:
    #use for tutors and timepoint from a juvenile
    def __init__(self, song_dir):
        self.song_dir = song_dir
        [_, self.id] = os.path.split(song_dir)
        self.embeddings_dir = os.path.join(self.song_dir, 'Embeddings')
        self.segmentation_dir = os.path.join(self.song_dir, 'Segmentations')

class Juvenile:
    #use to contain all tutor directories
    def __init__(self, juv_dir):
        self.timepoints = []
        for dir in os.listdir(juv_dir):
            if os.path.isdir(os.path.join(juv_dir, dir)):
                self.timepoints.append(Bird(dir))

class Pair:
    #contains tutor, pupil, out
    def __init__(self, pair_dir):
        [_, self.pair_id] = os.path.split(pair_dir)
        self.juveniles = []
        self.adults = []

        dirs = os.listdir(pair_dir)
        for dir in dirs:
            if "tutor" in dir: tutor_dir = os.path.join(pair_dir, dir)
            if "juv" in dir or "pupil" in dir: juv_dir = os.path.join(pair_dir, dir)
            else: self.out_dir = os.path.join(pair_dir, dir)

        for juv in os.listdir(juv_dir):
            juv_path = os.path.join(juv_dir, juv)
            if os.path.isdir(juv_path): self.juveniles.append(Juvenile(juv_path))
        for tutor in os.listdir(tutor_dir):
            tutor_path = os.path.join(tutor_dir, tutor)
            if os.path.isdir(tutor_path): self.adults.append(Bird(tutor_path))

        self.kl_df = pd.DataFrame("Pupil", "Tutor", "D_KL(P|T)")
        self.df_path = os.path.join(self.out_dir, f"{self.pair_id}.csv")


def run_dkl_one_pair(pair_dir):
    pair = Pair(pair_dir)
    for juv in pair.juveniles:
        for timepoint in juv.timepoints:
            P = timepoint.embeddings_dir
            for tutor in pair.adults:
                T = tutor.embeddings_dir
                kl = dkl.run_dkl(P, T, n_mc_samples = 1000)
                pair.kl_df.loc[len(pair.kl_df)] = [timepoint.id, tutor.id, kl]
    pair.kl_df.to_csv(pair.df_path)

def run_dkl_multiple_pairs(master_dir):
    for pair in os.listdir(master_dir):
        pair_path = os.path.join(master_dir, pair)
        if os.path.isdir(pair_path):
            run_dkl_one_pair(pair_path)

if __name__ == "__main__":
    master_dir = ""
    run_dkl_multiple_pairs(master_dir)

