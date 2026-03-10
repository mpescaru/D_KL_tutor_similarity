# D_KL_tutor_similarity
Project includes functions for calculating K-L divergence from a pupil's syllable populations to a father's syllable population. 

Functions in retutoring_process are tailored for the following directory structure: 

```
pair_x
|--- pair_x_tutors
|      |--- tutor1
|      |--- tutor2
|      |--- ...
|
|--- pair_x_pupils
|      |--- pupil1
|      |      |--- timepoint1
|      |      |--- timepoint2
|      |      |--- ...
|      |--- pupil2
|      |      |--- timepoint1
|      |      |--- timepoint1
|      |      |--- ...
|      |--- ...
|--- pair_x_out

```

This is useful for comparing how well juveniles immitates several possible tutors at different ages. 

Code requirements: avn, scikit, pandas, numpy, os
