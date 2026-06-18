from __future__ import annotations

import copy

from tclkg import generator_common
from tclkg import ground_truth
from tclkg import qcn_generator2 as qgen
from tclkg.allen_list import ALLEN_RELATIONS


MINI_KG = "MiniKG"


def test_mini_end_to_end_learner_matches_snapshot(tmp_path, monkeypatch) -> None:
    kg_dir = tmp_path / MINI_KG
    kg_dir.mkdir()
    (kg_dir / "data.quintuplet").write_text(
        "\n".join(
            [
                "e1\thttp://www.wikidata.org/prop/P1\tv1\t2000-01-01\t2000-01-02",
                "e1\thttp://www.wikidata.org/prop/P2\tv2\t2000-01-03\t2000-01-04",
                "e1\thttp://www.wikidata.org/prop/P3\tv3\t2000-01-05\t2000-01-06",
                "e2\thttp://www.wikidata.org/prop/P1\tv1\t2001-01-01\t2001-01-02",
                "e2\thttp://www.wikidata.org/prop/P2\tv2\t2001-01-02\t2001-01-03",
                "e2\thttp://www.wikidata.org/prop/P3\tv3\t2001-01-04\t2001-01-05",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(generator_common, "ROOT_DATA", tmp_path)

    entities, properties = qgen.read_KG(MINI_KG, print_properties_count=False)
    network = qgen.build_entity_network(
        properties=properties,
        default_constraints=ALLEN_RELATIONS,
        top_k_fraction=1.0,
        entities=entities,
    )
    kept_properties = sorted({property_ for pair in network for property_ in pair})

    result = qgen.learner2(copy.deepcopy(network), entities, kept_properties)

    assert result is not None
    assert (
        ground_truth.canonical_section_hash(result["initial"])
        == "ba32ab731c95136b3c04f2a0bf08e4278c0341a4d3976795305c55ed95d09433"
    )
    assert (
        ground_truth.canonical_section_hash(result["after_oracle"])
        == "6597418d9171fec0ea27257ca94abd19ced763e109b8118ddba456c270fcfdc9"
    )
    assert (
        ground_truth.canonical_section_hash(result["after_propagation"])
        == "5c79c8f2d33a98c9d68fcf25dd848d00564b561c8c2c4610d1f81ae2b365a278"
    )

    assert ground_truth.semantic_stats(result["after_oracle"]) == {
        "domains": 6,
        "relations": 78,
        "status_counts": {"None": 70, "Observed": 8},
        "support_counts": {"negative": 0, "none": 0, "positive": 8, "zero": 70},
        "domains_with_observed": 6,
        "domains_with_positive_support": 6,
        "positive_relations_per_domain": {
            "min": 1,
            "max": 2,
            "avg": 1.3333333333333333,
        },
    }
    assert ground_truth.semantic_stats(result["after_propagation"]) == {
        "domains": 6,
        "relations": 8,
        "status_counts": {"Observed": 8},
        "support_counts": {"negative": 0, "none": 0, "positive": 8, "zero": 0},
        "domains_with_observed": 6,
        "domains_with_positive_support": 6,
        "positive_relations_per_domain": {
            "min": 1,
            "max": 2,
            "avg": 1.3333333333333333,
        },
    }
