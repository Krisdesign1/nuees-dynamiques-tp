import numpy as np
import pytest

from src import AlternativeDynamicClouds
from src.data import make_demo_data


def test_all_representations_produce_non_empty_clusters():
    X, _ = make_demo_data(n_per_cluster=30, random_state=11)
    for rep in ["point", "points", "axes", "distribution", "structure"]:
        model = AlternativeDynamicClouds(
            n_clusters=3,
            representation=rep,
            n_prototypes=4,
            n_axes=1,
            n_init=2,
            random_state=7,
            max_iter=50,
        ).fit(X)
        assert model.labels_.shape == (len(X),)
        counts = np.bincount(model.labels_, minlength=3)
        assert np.all(counts > 0)
        assert len(model.representatives_) == 3


def test_predict_matches_training_assignment_for_point_case():
    X, _ = make_demo_data(n_per_cluster=25, random_state=5)
    model = AlternativeDynamicClouds(
        n_clusters=3, representation="point", n_init=3, random_state=3
    ).fit(X)
    pred = model.predict(X)
    assert np.array_equal(pred, model.labels_)


def test_reproducibility_with_fixed_seed():
    X, _ = make_demo_data(n_per_cluster=20, random_state=9)
    a = AlternativeDynamicClouds(n_clusters=3, representation="points", n_init=3, random_state=99).fit_predict(X)
    b = AlternativeDynamicClouds(n_clusters=3, representation="points", n_init=3, random_state=99).fit_predict(X)
    assert np.array_equal(a, b)


def test_invalid_nan_data_is_rejected():
    X = np.array([[0.0, 1.0], [np.nan, 2.0], [4.0, 5.0]])
    with pytest.raises(ValueError):
        AlternativeDynamicClouds(n_clusters=2).fit(X)
