from __future__ import annotations

import numpy as np

from tclkg import ground_truth
from tclkg import qcn_generator2 as qgen
from tclkg.allen_list import ALLEN_RELATIONS
from tclkg.generator_common import compose_allen_scores, converse_domain_scores
from tclkg.time_package import Entity, Interval, Triple


def _triple(head: str, relation: str, start: str, end: str) -> Triple:
    return Triple(
        head,
        relation,
        "value",
        Interval(np.datetime64(start, "D"), np.datetime64(end, "D")),
    )


def _entity(name: str, triples: list[Triple]) -> Entity:
    entity = Entity(name, np.datetime64("2023-12-31", "D"), "D")
    for triple in triples:
        entity.add_triple(triple)
    return entity


def _empty_domain() -> dict[str, object]:
    return qgen.with_relation_data(
        {relation: None for relation in ALLEN_RELATIONS},
        {relation: None for relation in ALLEN_RELATIONS},
    )


def _observed_domain(relation: str, support: float = 1.0) -> dict[str, object]:
    return qgen.with_relation_data(
        {relation: support},
        {relation: qgen.STATUS_OBSERVED},
    )


def test_query_computes_expected_support_ratios() -> None:
    p1 = "http://www.wikidata.org/prop/P1"
    p2 = "http://www.wikidata.org/prop/P2"
    entities = {
        "e1": _entity(
            "e1",
            [
                _triple("e1", p1, "2000-01-01", "2000-01-02"),
                _triple("e1", p2, "2000-01-03", "2000-01-04"),
            ],
        ),
        "e2": _entity(
            "e2",
            [
                _triple("e2", p1, "2001-01-02", "2001-01-03"),
                _triple("e2", p2, "2001-01-01", "2001-01-02"),
            ],
        ),
    }

    result = qgen.Query((p1, p2), entities)
    scores = qgen.relation_scores(result)
    statuses = qgen.relation_statuses(result)

    assert scores["before"] == 0.5
    assert scores["met_by"] == 0.5
    assert scores["after"] == 0.0
    assert statuses["before"] == qgen.STATUS_OBSERVED
    assert statuses["met_by"] == qgen.STATUS_OBSERVED
    assert statuses["after"] is None


def test_converse_and_composition_preserve_expected_relations() -> None:
    domain = _observed_domain("before", 0.75)
    converse = converse_domain_scores(domain)
    composition = compose_allen_scores({"before": 0.75}, {"meets": 0.5})

    assert qgen.relation_scores(converse) == {"after": 0.75}
    assert qgen.relation_statuses(converse) == {"after": qgen.STATUS_OBSERVED}
    assert composition == {"before": 0.5}


def test_propagate_and_filter_infers_missing_composition() -> None:
    a = "http://www.wikidata.org/prop/PA"
    b = "http://www.wikidata.org/prop/PB"
    c = "http://www.wikidata.org/prop/PC"
    qcn = {
        (a, b): _observed_domain("before"),
        (b, a): _observed_domain("after"),
        (b, c): _observed_domain("before"),
        (c, b): _observed_domain("after"),
        (a, c): _empty_domain(),
        (c, a): _empty_domain(),
    }

    result = qgen.propagateAndFilter(qcn)

    assert qgen.relation_scores(result[(a, c)]) == {"before": 1.0}
    assert qgen.relation_statuses(result[(a, c)]) == {"before": qgen.STATUS_INFERRED}
    assert qgen.relation_scores(result[(c, a)]) == {"after": 1.0}
    assert qgen.relation_statuses(result[(c, a)]) == {"after": qgen.STATUS_INFERRED}


def test_ground_truth_semantic_stats_and_transition_stats() -> None:
    after_oracle = {
        "a___b": _observed_domain("before"),
        "a___c": _empty_domain(),
    }
    after_propagation = {
        "a___b": _observed_domain("before"),
        "a___c": qgen.with_relation_data(
            {"before": 1.0},
            {"before": qgen.STATUS_INFERRED},
        ),
    }

    stats = ground_truth.semantic_stats(after_oracle)
    transitions = ground_truth.transition_stats(after_oracle, after_propagation)

    assert stats["domains"] == 2
    assert stats["relations"] == 14
    assert stats["status_counts"] == {"None": 13, "Observed": 1}
    assert stats["support_counts"] == {
        "negative": 0,
        "none": 13,
        "positive": 1,
        "zero": 0,
    }
    assert transitions["none_after_oracle_total"] == 13
    assert transitions["none_to_inferred"] == 1
    assert transitions["domains_without_positive_to_positive"] == 1
