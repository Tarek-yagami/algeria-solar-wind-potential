from src.clustering import CLUSTER_FEATURES, CLUSTER_NAMES, N_CLUSTERS


def test_cluster_features_has_no_duplicates():
    assert len(CLUSTER_FEATURES) == len(set(CLUSTER_FEATURES))


def test_cluster_names_cover_every_cluster_index():
    assert set(CLUSTER_NAMES.keys()) == set(range(N_CLUSTERS))


def test_cluster_names_are_unique():
    assert len(CLUSTER_NAMES.values()) == len(set(CLUSTER_NAMES.values()))
