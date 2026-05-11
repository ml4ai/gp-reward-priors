import h5py
import numpy as np

with h5py.File("../data/bb/bbway_tuning_set.hdf5", "a") as f:
    with h5py.File("../data/bb/bbway.hdf5") as g:
        obs = np.concatenate(
            [g["states"], g["actions"]],
            axis=-1,
        )

        f.create_dataset("obs", data=obs[:, :-1, :].reshape(-1, obs.shape[-1]))
        f.create_dataset(
            "aux_obs", data=g["states"][:, 1:, :].reshape(-1, g["states"].shape[-1])
        )