import numpy as np

from src.dynamic_clouds import DynamicClouds
from src.data import make_demo_dataset, standardize


def test_all_representations_run_and_produce_k_nonempty_clusters():
    X, _ = make_demo_dataset(n_per_cluster=30, random_state=4)
    X, _, _ = standardize(X)
    for kind in ["point", "points", "axes", "distribution", "structure"]:
        model = DynamicClouds(
            n_clusters=3,
            representation=kind,
            n_representatives=4,
            n_axes=1,
            max_iter=30,
            random_state=1,
        ).fit(X)
        assert model.labels_.shape == (len(X),)
        counts = np.bincount(model.labels_, minlength=3)
        assert np.all(counts > 0)
        assert len(model.representatives_) == 3


def test_point_representation_predict():
    X = np.array([[0.0, 0.0], [0.1, 0.2], [5.0, 5.0], [5.2, 5.1]])
    model = DynamicClouds(n_clusters=2, representation="point", random_state=0).fit(X)
    pred = model.predict(np.array([[0.05, 0.0], [5.1, 5.0]]))
    assert pred[0] != pred[1]
