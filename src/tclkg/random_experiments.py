import argparse
import copy
import json
from collections.abc import Mapping
from collections import deque
import os
from itertools import combinations
import multiprocessing as mp
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from . import allen_relations as AllenR
from .allen_list import ALLEN_RELATIONS
from .generator_common import (
    RESULTS_ROOT as results_root,
    build_complete_network as build_entity_network,
    compose_allen_set as composition_Allen,
    converse_domain_set as converse_domain,
    converse_domain_scores,
    path_consistency_classique,
    read_kg as read_KG,
)


'''
This module is for running some axperiments
'''



def read_qcn_from_json(jsonfile: str) -> dict[tuple[str, str], dict[str, float]]:
    '''
    Read a QCN from a JSON file and return it as a dictionary.
    The JSON file should have the following structure:
    {
        "after_oracle": {
            "p1___p2": {"before": 1.0, "meets": 0.9},
            "p2___p1": {"after": 1.0, "met_by": 0.9},
            ...
        }
    }
    '''
    json_path = Path(jsonfile)
    if not json_path.is_absolute():
        json_path = results_root / json_path

    if not json_path.exists():
        raise FileNotFoundError(f"QCN JSON file not found: {json_path}")

    with json_path.open("r", encoding="UTF-8") as handle:
        qcn_data = json.load(handle)

    after_oracle = qcn_data.get("after_oracle")
    if after_oracle is None:
        raise ValueError(f"Missing 'after_oracle' section in {json_path}")

    qcn: dict[tuple[str, str], dict[str, float]] = {}
    for pair_key, raw_domain in after_oracle.items():
        if "___" not in pair_key:
            raise ValueError(f"Invalid QCN pair key: {pair_key}")
        property1, property2 = pair_key.split("___", 1)

        if isinstance(raw_domain, list):
            domain = {relation: 0.0 for relation in raw_domain}
        elif isinstance(raw_domain, dict):
            domain = {
                relation: float(score)
                for relation, score in raw_domain.items()
            }
        else:
            raise TypeError(
                f"Unsupported domain type for pair {pair_key}: {type(raw_domain).__name__}"
            )

        qcn[(property1, property2)] = domain

    return qcn


def domainswith1relations(jsonfile: str) -> Path:
    '''
    Create a text file listing each pair whose domain contains relations with score 1.0,
    alongside the number of such relations.
    '''
    qcn = read_qcn_from_json(jsonfile)

    json_path = Path(jsonfile)
    report_name = f"{json_path.stem}_domains_with_score_1.txt"
    report_path = results_root / report_name

    lines: list[str] = []
    total_pairs = 0
    for (property1, property2), domain in qcn.items():
        nb_relations_score_1 = sum(1 for score in domain.values() if score == 1.0)
        if nb_relations_score_1 > 0:
            total_pairs += 1
            lines.append(
                f"{property1}___{property2}\t{nb_relations_score_1}"
            )

    header = [
        "pair\tnumber_of_relations_with_score_1.0",
        f"total_pairs\t{total_pairs}",
        "",
    ]
    report_path.write_text("\n".join(header + lines) + "\n", encoding="UTF-8")
    return report_path


def _prepare_qcn_for_threshold(
    qcn: dict[tuple[str, str], dict[str, float]], threshold: float
) -> dict[tuple[str, str], dict[str, float]]:
    '''
    Ensure each domain has at least one relation with score >= threshold,
    otherwise reset it to all Allen relations with score 0.0.
    '''
    for _, domain in qcn.items():
        if not any(score >= threshold for score in domain.values()):
            for relation in ALLEN_RELATIONS:
                domain[relation] = 0.0
    return qcn


def _format_domain(domain: dict[str, float]) -> str:
    if not domain:
        return "{}"
    ordered = sorted(domain.items(), key=lambda item: item[0])
    return "{" + ", ".join(f"{rel}:{score:.6g}" for rel, score in ordered) + "}"


def incoherence_report(
    jsonfile: str,
    threshold: float = 1.0,
    max_triplets: int = 300,
    trace_tail: int = 30,
) -> Path:
    '''
    Generate a report of contradictory triplets and the propagation path that
    leads to the first detected collapse during path consistency.
    '''
    initial_qcn = read_qcn_from_json(jsonfile)
    qcn = _prepare_qcn_for_threshold(copy.deepcopy(initial_qcn), threshold)
    properties = sorted({p for (x, y) in qcn.keys() for p in (x, y)})

    contradictory_triplets: list[tuple[str, str, str, set[str], set[str], set[str]]] = []
    for i in properties:
        for j in properties:
            if j == i:
                continue
            for k in properties:
                if k == i or k == j:
                    continue
                if (i, j) not in qcn or (j, k) not in qcn or (i, k) not in qcn:
                    continue
                domain_ij = set(qcn[(i, j)].keys())
                domain_jk = set(qcn[(j, k)].keys())
                domain_ik = set(qcn[(i, k)].keys())
                composed = composition_Allen(domain_ij, domain_jk)
                if not (domain_ik & composed):
                    contradictory_triplets.append((i, j, k, domain_ij, domain_jk, domain_ik))
                    if len(contradictory_triplets) >= max_triplets:
                        break
            if len(contradictory_triplets) >= max_triplets:
                break
        if len(contradictory_triplets) >= max_triplets:
            break

    qcn_pc = copy.deepcopy(qcn)
    queue = deque(qcn_pc.keys())
    trace_events: list[str] = []
    collapse_info: dict[str, Any] | None = None
    step = 0

    while queue and collapse_info is None:
        i, j = queue.popleft()
        for k in properties:
            if k == i or k == j:
                continue
            if (i, k) not in qcn_pc or (k, j) not in qcn_pc:
                raise ValueError(
                    f"Missing arc in qcn_copy during propagation: {(i, k)} or {(k, j)}"
                )

            step += 1

            dik = qcn_pc[(i, k)]
            composed_ij_jk = composition_Allen(set(qcn_pc[(i, j)]), set(qcn_pc[(j, k)]))
            new_dik = {relation: 0.0 for relation in dik if relation in composed_ij_jk}

            if new_dik != dik:
                if not new_dik:
                    collapse_info = {
                        "collapsed_pair": (i, k),
                        "path": (i, j, k),
                        "left_pair": (i, j),
                        "right_pair": (j, k),
                        "target_before": dict(dik),
                        "composed": sorted(composed_ij_jk),
                        "step": step,
                    }
                    break
                qcn_pc[(i, k)] = new_dik
                qcn_pc[(k, i)] = converse_domain_scores(new_dik)
                queue.append((i, k))
                trace_events.append(
                    f"step {step}: reduce ({i}, {k}) via ({i}, {j}) o ({j}, {k}) | "
                    f"before={sorted(dik.keys())} after={sorted(new_dik.keys())}"
                )

            dkj = qcn_pc[(k, j)]
            composed_ki_ij = composition_Allen(set(qcn_pc[(k, i)]), set(qcn_pc[(i, j)]))
            new_dkj = {relation: 0.0 for relation in dkj if relation in composed_ki_ij}

            if new_dkj != dkj:
                if not new_dkj:
                    collapse_info = {
                        "collapsed_pair": (k, j),
                        "path": (k, i, j),
                        "left_pair": (k, i),
                        "right_pair": (i, j),
                        "target_before": dict(dkj),
                        "composed": sorted(composed_ki_ij),
                        "step": step,
                    }
                    break
                qcn_pc[(k, j)] = new_dkj
                qcn_pc[(j, k)] = converse_domain_scores(new_dkj)
                queue.append((k, j))
                trace_events.append(
                    f"step {step}: reduce ({k}, {j}) via ({k}, {i}) o ({i}, {j}) | "
                    f"before={sorted(dkj.keys())} after={sorted(new_dkj.keys())}"
                )

    json_path = Path(jsonfile)
    report_path = results_root / f"{json_path.stem}_incoherence_report.txt"

    lines: list[str] = [
        f"source_json\t{jsonfile}",
        f"threshold\t{threshold}",
        f"contradictory_triplets_found\t{len(contradictory_triplets)}",
        f"triplets_truncated\t{len(contradictory_triplets) >= max_triplets}",
        "",
        "[A] Contradictory triplets (initial domains after threshold preparation)",
    ]

    if contradictory_triplets:
        for idx, (i, j, k, _, _, _) in enumerate(contradictory_triplets, 1):
            lines.append(f"{idx}. path: {i} -> {j} -> {k}")
            lines.append(f"   domain({i}, {j}) initial: {_format_domain(initial_qcn[(i, j)])}")
            lines.append(f"   domain({j}, {k}) initial: {_format_domain(initial_qcn[(j, k)])}")
            lines.append(f"   domain({i}, {k}) initial: {_format_domain(initial_qcn[(i, k)])}")
            lines.append("")
    else:
        lines.append("No contradictory triplet found at initialization.")
        lines.append("")

    lines.append("[B] Path consistency collapse path")
    if collapse_info is None:
        lines.append("No collapse detected during path consistency propagation.")
    else:
        p1, p2, p3 = collapse_info["path"]
        c1, c2 = collapse_info["collapsed_pair"]
        l1, l2 = collapse_info["left_pair"]
        r1, r2 = collapse_info["right_pair"]
        lines.append(f"collapse_step: {collapse_info['step']}")
        lines.append(f"collapse_pair: ({c1}, {c2})")
        lines.append(f"incoherence_path: {p1} -> {p2} -> {p3}")
        lines.append(f"left_arc: ({l1}, {l2})")
        lines.append(f"right_arc: ({r1}, {r2})")
        lines.append(
            "composed_relations(left_arc o right_arc): "
            + str(collapse_info["composed"])
        )
        lines.append(
            "target_domain_before_collapse: "
            + _format_domain(collapse_info["target_before"])
        )
        lines.append("")
        lines.append("Initial domains of collapse triplet:")
        lines.append(f"domain({l1}, {l2}) initial: {_format_domain(initial_qcn[(l1, l2)])}")
        lines.append(f"domain({r1}, {r2}) initial: {_format_domain(initial_qcn[(r1, r2)])}")
        lines.append(f"domain({c1}, {c2}) initial: {_format_domain(initial_qcn[(c1, c2)])}")
        lines.append("")
        lines.append(f"Recent propagation steps before collapse (last {trace_tail}):")
        if trace_events:
            lines.extend(trace_events[-trace_tail:])
        else:
            lines.append("No prior reduction before collapse.")

    report_path.write_text("\n".join(lines) + "\n", encoding="UTF-8")
    return report_path


def property_implied_quadruplets(kg_name: str) -> None:
    '''
    Plot a histogram: for each property, the number of quadruplets (triples) it is involved in.
    kg_name is the KG identifier (e.g. "Q6256") as used by read_KG.
    Properties are numbered 1..n (sorted by count desc). The plot is saved locally.
    '''
    entities, _ = read_KG(kg_name, False)

    histogram: dict[str, int] = {}
    for entity in entities.values():
        for prop, triples in entity.triples_per_p.items():
            histogram[prop] = histogram.get(prop, 0) + len(triples)

    sorted_items = sorted(histogram.items(), key=lambda x: -x[1])
    indices = list(range(1, len(sorted_items) + 1))
    counts = [item[1] for item in sorted_items]
    labels = [f"{i}. {item[0].split('/')[-1]}" for i, item in enumerate(sorted_items, 1)]

    fig, ax = plt.subplots(figsize=(max(10, len(indices) * 0.5), 6))
    ax.bar(indices, counts)
    ax.set_xticks(indices)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Property (ranked by quadruplet count)")
    ax.set_ylabel("Number of quadruplets (log scale)")
    ax.set_title(f"Quadruplets per property — {kg_name}")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    output_path = Path(__file__).resolve().parents[2] / "Results" / f"quadruplets_histogram_{kg_name}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


def _extract_supports_from_domain(raw_domain: Any) -> list[float]:
    """Extract non-None numeric supports in [0, 1] from one domain representation."""
    supports: list[float] = []

    if isinstance(raw_domain, Mapping):
        structured_relations = raw_domain.get("relations")
        if isinstance(structured_relations, Mapping):
            for payload in structured_relations.values():
                if isinstance(payload, Mapping):
                    support = payload.get("support")
                    if support is not None and isinstance(support, (int, float)) and 0.0 <= float(support) <= 1.0:
                        supports.append(float(support))
            return supports


        for value in raw_domain.values():
            if value is not None and isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0:
                supports.append(float(value))
            elif isinstance(value, Mapping):
                support = value.get("support")
                if support is not None and isinstance(support, (int, float)) and 0.0 <= float(support) <= 1.0:
                    supports.append(float(support))

    return supports


def _build_histogram_counts(values: list[float], bin_width: float) -> tuple[list[float], list[int]]:
    """Build histogram counts on [0, 1] with fixed-width bins."""
    nbins = int(round(1.0 / bin_width))
    edges = [round(i * bin_width, 10) for i in range(nbins + 1)]
    counts = [0 for _ in range(nbins)]

    for value in values:
        if value < 0.0 or value > 1.0:
            continue
        index = min(int(value / bin_width), nbins - 1)
        counts[index] += 1

    return edges, counts


def _extract_inferred_only_supports(
    oracle_domain: Any, propagation_domain: Any
) -> list[float]:
    """
    Extract supports of relations that exist only in propagation_domain
    (inferred only - not present in oracle_domain).
    """
    supports: list[float] = []
    if not isinstance(oracle_domain, Mapping) or not isinstance(propagation_domain, Mapping):
        return supports


    oracle_relations = set()
    propagation_relations = set()


    oracle_structured = oracle_domain.get("relations")
    if isinstance(oracle_structured, Mapping):
        oracle_relations.update(oracle_structured.keys())
    else:

        oracle_relations.update(k for k in oracle_domain.keys() if k != "relations")


    propagation_structured = propagation_domain.get("relations")
    if isinstance(propagation_structured, Mapping):
        for relation, payload in propagation_structured.items():
            if relation not in oracle_relations and isinstance(payload, Mapping):
                support = payload.get("support")
                if support is not None and isinstance(support, (int, float)) and 0.0 <= float(support) <= 1.0:
                    supports.append(float(support))
    else:

        for relation, value in propagation_domain.items():
            if relation == "relations":
                continue
            if relation not in oracle_relations:
                if value is not None and isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0:
                    supports.append(float(value))
                elif isinstance(value, Mapping):
                    support = value.get("support")
                    if support is not None and isinstance(support, (int, float)) and 0.0 <= float(support) <= 1.0:
                        supports.append(float(support))

    return supports


def _domain_has_positive_support(raw_domain: Any) -> bool:
    """Return True if a domain has at least one non-None support strictly > 0."""
    if not isinstance(raw_domain, Mapping):
        return False

    structured_relations = raw_domain.get("relations")
    if isinstance(structured_relations, Mapping):
        for payload in structured_relations.values():
            if not isinstance(payload, Mapping):
                continue
            support = payload.get("support")
            if support is not None and isinstance(support, (int, float)) and float(support) > 0.0:
                return True
        return False


    for value in raw_domain.values():
        if value is not None and isinstance(value, (int, float)) and float(value) > 0.0:
            return True
        if isinstance(value, Mapping):
            support = value.get("support")
            if support is not None and isinstance(support, (int, float)) and float(support) > 0.0:
                return True
    return False


def _count_positive_relations_in_domain(raw_domain: Any) -> int:
    """Count relations with non-None support strictly > 0 in one domain."""
    if not isinstance(raw_domain, Mapping):
        return 0

    count = 0
    structured_relations = raw_domain.get("relations")
    if isinstance(structured_relations, Mapping):
        for payload in structured_relations.values():
            if not isinstance(payload, Mapping):
                continue
            support = payload.get("support")
            if support is not None and isinstance(support, (int, float)) and float(support) > 0.0:
                count += 1
        return count


    for value in raw_domain.values():
        if value is not None and isinstance(value, (int, float)) and float(value) > 0.0:
            count += 1
        elif isinstance(value, Mapping):
            support = value.get("support")
            if support is not None and isinstance(support, (int, float)) and float(support) > 0.0:
                count += 1
    return count


def _extract_statuses_from_domain(raw_domain: Any) -> list[str]:
    """Extract relation statuses from one domain (structured and legacy-safe)."""
    statuses: list[str] = []
    if not isinstance(raw_domain, Mapping):
        return statuses

    structured_relations = raw_domain.get("relations")
    if isinstance(structured_relations, Mapping):
        for payload in structured_relations.values():
            if not isinstance(payload, Mapping):
                statuses.append("other")
                continue
            status = payload.get("status")
            if status is None:
                statuses.append("none")
            elif isinstance(status, str):
                normalized = status.strip().lower()
                if normalized == "observed":
                    statuses.append("observed")
                elif normalized in {"inferred", "infered"}:
                    statuses.append("inferred")
                elif normalized in {"observed and inferred", "inferred and observed"}:
                    statuses.append("observed_and_inferred")
                elif normalized == "repaired":
                    statuses.append("repaired")
                else:
                    statuses.append("other")
            else:
                statuses.append("other")
        return statuses


    if isinstance(raw_domain, Mapping):
        for _ in raw_domain.values():
            statuses.append("none")
    return statuses


def _normalize_transition_status(status: Any) -> str | None:
    """Normalize raw status labels to the four publication heatmap categories."""
    if status is None:
        return "none"
    if not isinstance(status, str):
        return None

    normalized = status.strip().lower()
    if normalized == "observed":
        return "observed"
    if normalized in {"inferred", "infered"}:
        return "inferred"
    if normalized in {"observed and inferred", "inferred and observed"}:
        return "observed_and_inferred"
    if normalized == "none":
        return "none"
    return None


def _extract_relation_status_map(raw_domain: Any) -> dict[str, str | None]:
    """Extract a relation -> normalized status mapping from one domain."""
    if not isinstance(raw_domain, Mapping):
        return {}

    structured_relations = raw_domain.get("relations")
    if isinstance(structured_relations, Mapping):
        status_map: dict[str, str | None] = {}
        for relation, payload in structured_relations.items():
            if isinstance(payload, Mapping):
                status_map[str(relation)] = _normalize_transition_status(payload.get("status"))
            else:
                status_map[str(relation)] = None
        return status_map

    return {
        str(relation): _normalize_transition_status(None)
        for relation in raw_domain.keys()
    }


def count_domains_without_positive_to_positive(
    after_oracle_qcn: Mapping[str, Any],
    after_propagation_qcn: Mapping[str, Any],
) -> dict[str, int | float]:
    """
    Count domains that have no positive support in after_oracle and gain
    at least one positive support in after_propagation.

    The two QCN mappings are expected to use the same pair keys
    (e.g. "p1___p2"), but missing keys in after_propagation are tolerated.
    """
    domains_without_positive_after_oracle = 0
    domains_became_positive_after_propagation = 0

    for pair_key, oracle_domain in after_oracle_qcn.items():
        oracle_has_positive = _domain_has_positive_support(oracle_domain)
        propagation_domain = after_propagation_qcn.get(pair_key, {})
        propagation_has_positive = _domain_has_positive_support(propagation_domain)

        if not oracle_has_positive:
            domains_without_positive_after_oracle += 1
            if propagation_has_positive:
                domains_became_positive_after_propagation += 1

    ratio = (
        domains_became_positive_after_propagation
        / domains_without_positive_after_oracle
        * 100.0
        if domains_without_positive_after_oracle > 0
        else 0.0
    )

    return {
        "domains_without_positive_after_oracle": domains_without_positive_after_oracle,
        "domains_became_positive_after_propagation": domains_became_positive_after_propagation,
        "percentage_became_positive": ratio,
    }


def compute_iswc_after_propagation_positive_relation_stats() -> dict[str, dict[str, float]]:
    """
    Compute min/max/average number of positive relations per domain in
    after_propagation, for each TKG under Results/ISWC.
    """
    iswc_root = results_root / "ISWC"
    if not iswc_root.exists():
        raise FileNotFoundError(f"Missing ISWC results directory: {iswc_root}")

    json_files = sorted(iswc_root.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under: {iswc_root}")

    counts_per_kg: dict[str, list[int]] = {}
    files_seen_per_kg: dict[str, int] = {}

    for json_path in json_files:
        rel_path = json_path.relative_to(iswc_root)
        kg_name = rel_path.parts[0] if len(rel_path.parts) >= 2 else json_path.stem

        with json_path.open("r", encoding="UTF-8") as handle:
            payload = json.load(handle)

        after_propagation = payload.get("after_propagation")
        if not isinstance(after_propagation, Mapping):
            continue

        counts_per_kg.setdefault(kg_name, [])
        files_seen_per_kg[kg_name] = files_seen_per_kg.get(kg_name, 0) + 1

        for domain in after_propagation.values():
            counts_per_kg[kg_name].append(_count_positive_relations_in_domain(domain))

    if not counts_per_kg:
        raise ValueError("No valid after_propagation sections found in ISWC JSON files")

    summary: dict[str, dict[str, float]] = {}
    for kg_name in sorted(counts_per_kg):
        domain_counts = counts_per_kg[kg_name]
        n = len(domain_counts)
        avg = (sum(domain_counts) / n) if n > 0 else 0.0

        summary[kg_name] = {
            "domains_count": float(n),
            "positive_relations_per_domain_min": float(min(domain_counts)) if n > 0 else 0.0,
            "positive_relations_per_domain_max": float(max(domain_counts)) if n > 0 else 0.0,
            "positive_relations_per_domain_avg": avg,
            "json_files_seen": float(files_seen_per_kg.get(kg_name, 0)),
        }

    for kg_name in sorted(summary):
        stats = summary[kg_name]
        print(
            f"{kg_name}: min={int(stats['positive_relations_per_domain_min'])} "
            f"max={int(stats['positive_relations_per_domain_max'])} "
            f"moy={stats['positive_relations_per_domain_avg']:.4f} "
            f"(domains={int(stats['domains_count'])})"
        )

    return summary


def plot_iswc_domain_size_subplots() -> dict[str, dict[str, Any]]:
    """
    Generate, for each TKG under Results/ISWC, a 2-subplot figure with:
    1) Distribution of domain sizes for domains that contain at least one
       observed relation in after_propagation.
     2) Distribution of inferred-only domain sizes in after_propagation,
         restricted to domains with no observed information.

    A domain size is the number of relations in the domain.
    Inferred-only size counts relations with status "inferred" in
    after_propagation that are absent (or status none) in after_oracle.
    """
    iswc_root = results_root / "ISWC"
    if not iswc_root.exists():
        raise FileNotFoundError(f"Missing ISWC results directory: {iswc_root}")

    json_files = sorted(iswc_root.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under: {iswc_root}")

    observed_sizes_per_kg: dict[str, list[int]] = {}
    inferred_only_sizes_per_kg: dict[str, list[int]] = {}
    files_seen_per_kg: dict[str, int] = {}

    for json_path in json_files:
        rel_path = json_path.relative_to(iswc_root)
        kg_name = rel_path.parts[0] if len(rel_path.parts) >= 2 else json_path.stem

        with json_path.open("r", encoding="UTF-8") as handle:
            payload = json.load(handle)

        after_oracle = payload.get("after_oracle")
        after_propagation = payload.get("after_propagation")
        if not isinstance(after_oracle, Mapping) or not isinstance(after_propagation, Mapping):
            continue

        observed_sizes_per_kg.setdefault(kg_name, [])
        inferred_only_sizes_per_kg.setdefault(kg_name, [])
        files_seen_per_kg[kg_name] = files_seen_per_kg.get(kg_name, 0) + 1

        for pair_key, propagation_domain in after_propagation.items():
            if not isinstance(propagation_domain, Mapping):
                continue

            propagation_relations = propagation_domain.get("relations")
            if not isinstance(propagation_relations, Mapping):
                continue

            oracle_domain = after_oracle.get(pair_key, {})
            oracle_relations = (
                oracle_domain.get("relations", {})
                if isinstance(oracle_domain, Mapping)
                else {}
            )
            if not isinstance(oracle_relations, Mapping):
                oracle_relations = {}

            domain_size = len(propagation_relations)
            has_observed = False
            inferred_only_size = 0

            for relation, prop_payload in propagation_relations.items():
                if not isinstance(prop_payload, Mapping):
                    continue

                prop_status = _normalize_transition_status(prop_payload.get("status"))
                if prop_status in {"observed", "observed_and_inferred"}:
                    has_observed = True

                oracle_payload = oracle_relations.get(relation)
                oracle_status = (
                    _normalize_transition_status(oracle_payload.get("status"))
                    if isinstance(oracle_payload, Mapping)
                    else None
                )
                if prop_status == "inferred" and (
                    (not isinstance(oracle_payload, Mapping))
                    or oracle_status in {None, "none"}
                ):
                    inferred_only_size += 1

            if has_observed:
                observed_sizes_per_kg[kg_name].append(domain_size)
            else:
                inferred_only_sizes_per_kg[kg_name].append(inferred_only_size)

    if not observed_sizes_per_kg and not inferred_only_sizes_per_kg:
        raise ValueError("No valid after_oracle/after_propagation data found in ISWC JSON files")

    summary: dict[str, dict[str, Any]] = {}

    all_kgs = sorted(set(observed_sizes_per_kg.keys()) | set(inferred_only_sizes_per_kg.keys()))
    for kg_name in all_kgs:
        observed_sizes = observed_sizes_per_kg.get(kg_name, [])
        inferred_only_sizes = inferred_only_sizes_per_kg.get(kg_name, [])

        max_size = 0
        if observed_sizes:
            max_size = max(max_size, max(observed_sizes))
        if inferred_only_sizes:
            max_size = max(max_size, max(inferred_only_sizes))
        x_values = list(range(max_size + 1))

        observed_hist = [0] * (max_size + 1)
        inferred_only_hist = [0] * (max_size + 1)

        for size in observed_sizes:
            if 0 <= size <= max_size:
                observed_hist[size] += 1
        for size in inferred_only_sizes:
            if 0 <= size <= max_size:
                inferred_only_hist[size] += 1

        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 5.5))

        ax_left.bar(x_values, observed_hist, color="#2E86AB", width=0.85)
        ax_left.set_title(f"{kg_name} - Observed domains")
        ax_left.set_xlabel("Domain size (#relations)")
        ax_left.set_ylabel("Number of domains")
        ax_left.set_yscale("log")
        ax_left.set_xticks(x_values)
        ax_left.grid(axis="y", linestyle="--", alpha=0.5)

        ax_right.bar(x_values, inferred_only_hist, color="#2CA02C", width=0.85)
        ax_right.set_title(f"{kg_name} - Inferred-only domains")
        ax_right.set_xlabel("Inferred-only size (#relations)")
        ax_right.set_ylabel("Number of domains")
        ax_right.set_yscale("log")
        ax_right.set_xticks(x_values)
        ax_right.grid(axis="y", linestyle="--", alpha=0.5)

        plt.suptitle(f"{kg_name} - Domain size distributions", fontsize=13, fontweight="bold")
        plt.tight_layout()

        output_path = iswc_root / kg_name / f"domain_size_subplots_{kg_name}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)

        print(f"ISWC domain-size subplots saved to {output_path}")

        summary[kg_name] = {
            "observed_domain_sizes_count": len(observed_sizes),
            "inferred_only_domain_sizes_count": len(inferred_only_sizes),
            "observed_min": min(observed_sizes) if observed_sizes else 0,
            "observed_max": max(observed_sizes) if observed_sizes else 0,
            "observed_avg": (sum(observed_sizes) / len(observed_sizes)) if observed_sizes else 0.0,
            "inferred_only_min": min(inferred_only_sizes) if inferred_only_sizes else 0,
            "inferred_only_max": max(inferred_only_sizes) if inferred_only_sizes else 0,
            "inferred_only_avg": (sum(inferred_only_sizes) / len(inferred_only_sizes)) if inferred_only_sizes else 0.0,
            "json_files_seen": files_seen_per_kg.get(kg_name, 0),
        }

    return summary


def plot_iswc_file_domain_size_histograms() -> dict[str, dict[str, Any]]:
    """
    For each JSON file under Results/ISWC, build one histogram of
    after_propagation domain sizes with bins 0..13.

    Bars represent all domains. A green hatched overlay marks domains that are
    inferred-only (no observed information and at least one inferred relation).

    Output file is saved next to each JSON input as:
    <json_stem>_domain_size_histogram.png
    """
    iswc_root = results_root / "ISWC"
    if not iswc_root.exists():
        raise FileNotFoundError(f"Missing ISWC results directory: {iswc_root}")

    json_files = sorted(iswc_root.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under: {iswc_root}")

    label_by_kg = {
        "Q6256": "Country",
        "Q215380": "Musical group",
        "Q82955": "Politician",
    }

    summary: dict[str, dict[str, Any]] = {}
    x_values = list(range(14))

    for json_path in json_files:
        with json_path.open("r", encoding="UTF-8") as handle:
            payload = json.load(handle)

        after_propagation = payload.get("after_propagation")
        if not isinstance(after_propagation, Mapping):
            continue

        total_hist = [0] * 14
        inferred_only_hist = [0] * 14

        for domain in after_propagation.values():
            if not isinstance(domain, Mapping):
                continue
            relations = domain.get("relations")
            if not isinstance(relations, Mapping):
                continue

            domain_size = len(relations)
            size_bin = domain_size if 0 <= domain_size <= 13 else 13
            total_hist[size_bin] += 1

            has_observed = False
            has_inferred = False
            for payload_rel in relations.values():
                if not isinstance(payload_rel, Mapping):
                    continue
                status = _normalize_transition_status(payload_rel.get("status"))
                if status in {"observed", "observed_and_inferred"}:
                    has_observed = True
                if status == "inferred":
                    has_inferred = True

            if (not has_observed) and has_inferred:
                inferred_only_hist[size_bin] += 1

        fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
        ax.bar(x_values, total_hist, width=0.85, color="#2E86AB", label="all domains")
        ax.bar(
            x_values,
            inferred_only_hist,
            width=0.85,
            facecolor="none",
            edgecolor="green",
            hatch="\\\\\\\\",
            linewidth=1.0,
            label="inferred-only domains",
        )

        rel_path = json_path.relative_to(iswc_root)
        kg_id = rel_path.parts[0] if len(rel_path.parts) >= 2 else json_path.stem
        kg_label = label_by_kg.get(kg_id, kg_id)
        ax.set_title(f"{kg_label} ({kg_id}) - after_propagation domain-size distribution")
        ax.set_xlabel("Domain size (#relations)")
        ax.set_ylabel("Number of domains")
        ax.set_yscale("log")
        ax.set_xticks(x_values)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend()

        plt.tight_layout()
        output_path = json_path.with_name(f"{json_path.stem}_domain_size_histogram.png")
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"ISWC file histogram saved to {output_path}")

        summary[str(rel_path)] = {
            "total_domains": sum(total_hist),
            "inferred_only_domains": sum(inferred_only_hist),
            "total_hist": total_hist,
            "inferred_only_hist": inferred_only_hist,
            "output_file": str(output_path.relative_to(results_root.parent if results_root.parent.exists() else output_path.parent)),
        }

    return summary


def plot_iswc_status_transitions_stacked() -> dict[str, dict[str, dict[str, float]]]:
    """
    Plot stacked bar charts of status proportions for after_oracle and
    after_propagation, for each dataset under Results/ISWC.

    Returns per-dataset status proportions for both stages.
    """
    iswc_root = results_root / "ISWC"
    if not iswc_root.exists():
        raise FileNotFoundError(f"Missing ISWC results directory: {iswc_root}")

    json_files = sorted(iswc_root.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under: {iswc_root}")

    stages = ("after_oracle", "after_propagation")
    status_order = [
        "none",
        "observed",
        "inferred",
        "observed_and_inferred",
        "repaired",
        "other",
    ]
    status_labels = {
        "none": "none",
        "observed": "observed",
        "inferred": "inferred",
        "observed_and_inferred": "observed+inferred",
        "repaired": "repaired",
        "other": "other",
    }
    status_colors = {
        "none": "#9AA5B1",
        "observed": "#2E86AB",
        "inferred": "#F18F01",
        "observed_and_inferred": "#5A9367",
        "repaired": "#C73E1D",
        "other": "#6C5B7B",
    }

    counts_per_dataset: dict[str, dict[str, dict[str, int]]] = {}

    for json_path in json_files:
        rel_path = json_path.relative_to(iswc_root)
        dataset = rel_path.parts[0] if len(rel_path.parts) >= 2 else json_path.stem

        with json_path.open("r", encoding="UTF-8") as handle:
            payload = json.load(handle)

        counts_per_dataset.setdefault(
            dataset,
            {
                stage: {status: 0 for status in status_order}
                for stage in stages
            },
        )

        for stage in stages:
            stage_domains = payload.get(stage)
            if not isinstance(stage_domains, Mapping):
                continue
            for domain in stage_domains.values():
                for status in _extract_statuses_from_domain(domain):
                    if status not in counts_per_dataset[dataset][stage]:
                        counts_per_dataset[dataset][stage]["other"] += 1
                    else:
                        counts_per_dataset[dataset][stage][status] += 1

    if not counts_per_dataset:
        raise ValueError("No valid stage data found in ISWC JSON files")

    summary: dict[str, dict[str, dict[str, float]]] = {}

    for dataset in sorted(counts_per_dataset.keys()):
        dataset_counts = counts_per_dataset[dataset]
        stage_props: dict[str, dict[str, float]] = {}

        for stage in stages:
            total = sum(dataset_counts[stage].values())
            if total == 0:
                stage_props[stage] = {status: 0.0 for status in status_order}
            else:
                stage_props[stage] = {
                    status: (dataset_counts[stage][status] / total) * 100.0
                    for status in status_order
                }

        summary[dataset] = stage_props


        fig, ax = plt.subplots(figsize=(9, 5.5))
        x_labels = ["after_oracle", "after_propagation"]
        x_pos = [0, 1]
        bottoms = [0.0, 0.0]

        for status in status_order:
            values = [
                stage_props["after_oracle"][status],
                stage_props["after_propagation"][status],
            ]
            ax.bar(
                x_pos,
                values,
                bottom=bottoms,
                color=status_colors[status],
                label=status_labels[status],
                width=0.6,
            )
            bottoms = [bottoms[i] + values[i] for i in range(2)]

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_ylim(0.0, 100.0)
        ax.set_ylabel("Proportion of relations (%)")
        ax.set_xlabel("Stage")
        ax.set_title(f"{dataset} - Status transitions (stacked)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

        plt.tight_layout()
        output_path = iswc_root / dataset / f"status_transitions_stacked_{dataset}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Status transition plot saved to {output_path}")

    return summary


def plot_iswc_status_transition_heatmaps() -> dict[str, dict[str, Any]]:
    """
    Generate publication-ready heatmaps of relation-level status transitions
    from after_oracle to after_propagation for each ISWC dataset.

    Rows correspond to the status before propagation and columns to the
    status after propagation. Cell values are proportions of relations (%).
    """
    iswc_root = results_root / "ISWC"
    if not iswc_root.exists():
        raise FileNotFoundError(f"Missing ISWC results directory: {iswc_root}")

    json_files = sorted(iswc_root.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under: {iswc_root}")

    dataset_display_names = {
        "Q6256": "Country",
        "Q215380": "Musical Group",
        "Q82955": "Politician",
    }
    status_order = ["none", "observed", "inferred", "observed_and_inferred"]
    status_labels = ["none", "observed", "inferred", "observed+\ninferred"]
    status_to_index = {status: index for index, status in enumerate(status_order)}

    transition_counts_per_dataset: dict[str, list[list[int]]] = {}
    matched_relations_per_dataset: dict[str, int] = {}
    skipped_relations_per_dataset: dict[str, int] = {}

    for json_path in json_files:
        rel_path = json_path.relative_to(iswc_root)
        dataset = rel_path.parts[0] if len(rel_path.parts) >= 2 else json_path.stem

        with json_path.open("r", encoding="UTF-8") as handle:
            payload = json.load(handle)

        after_oracle = payload.get("after_oracle")
        after_propagation = payload.get("after_propagation")
        if not isinstance(after_oracle, Mapping) or not isinstance(after_propagation, Mapping):
            continue

        if dataset not in transition_counts_per_dataset:
            transition_counts_per_dataset[dataset] = [
                [0 for _ in status_order] for _ in status_order
            ]
            matched_relations_per_dataset[dataset] = 0
            skipped_relations_per_dataset[dataset] = 0

        pair_keys = set(after_oracle.keys()) | set(after_propagation.keys())
        for pair_key in pair_keys:
            before_statuses = _extract_relation_status_map(after_oracle.get(pair_key, {}))
            after_statuses = _extract_relation_status_map(after_propagation.get(pair_key, {}))

            relation_names = set(before_statuses.keys()) | set(after_statuses.keys())
            for relation_name in relation_names:
                before_status = before_statuses.get(relation_name, "none")
                after_status = after_statuses.get(relation_name, "none")

                if before_status not in status_to_index or after_status not in status_to_index:
                    skipped_relations_per_dataset[dataset] += 1
                    continue

                transition_counts_per_dataset[dataset][status_to_index[before_status]][status_to_index[after_status]] += 1
                matched_relations_per_dataset[dataset] += 1

    if not transition_counts_per_dataset:
        raise ValueError("No valid after_oracle/after_propagation sections found in ISWC JSON files")

    summary: dict[str, dict[str, Any]] = {}
    rc_params = {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }

    preferred_order = ["Q6256", "Q215380", "Q82955"]
    ordered_datasets = [
        dataset for dataset in preferred_order if dataset in transition_counts_per_dataset
    ]
    ordered_datasets.extend(
        sorted(
            dataset
            for dataset in transition_counts_per_dataset.keys()
            if dataset not in ordered_datasets
        )
    )

    for dataset in ordered_datasets:
        counts = transition_counts_per_dataset[dataset]
        matched_relations = matched_relations_per_dataset[dataset]
        skipped_relations = skipped_relations_per_dataset[dataset]
        if matched_relations == 0:
            percentages = [[0.0 for _ in status_order] for _ in status_order]
        else:
            percentages = [
                [cell_count * 100.0 / matched_relations for cell_count in row]
                for row in counts
            ]

        summary[dataset] = {
            "percentages": percentages,
            "counts": counts,
            "matched_relations": matched_relations,
            "skipped_relations": skipped_relations,
        }

    global_vmax = max(
        max(max(row) for row in summary[dataset]["percentages"])
        for dataset in ordered_datasets
    )
    if global_vmax <= 0.0:
        global_vmax = 1.0

    with plt.rc_context(rc_params):
        fig, axes = plt.subplots(
            1,
            len(ordered_datasets),
            figsize=(4.2 * len(ordered_datasets), 4),
            constrained_layout=True,
            sharey=True,
        )
        if len(ordered_datasets) == 1:
            axes = [axes]

        image = None
        for index, dataset in enumerate(ordered_datasets):
            ax = axes[index]
            display_name = dataset_display_names.get(dataset, dataset)
            percentages = summary[dataset]["percentages"]

            image = ax.imshow(percentages, cmap="Blues", vmin=0.0, vmax=global_vmax)
            ax.set_xticks(range(len(status_labels)))
            ax.set_xticklabels(status_labels, rotation=0, ha="center")
            ax.set_title(display_name, pad=8)

            ax.set_yticks(range(len(status_labels)))
            if index == 0:
                ax.set_yticklabels(status_labels)
                ax.set_ylabel("Before propagation")
                ax.tick_params(axis="y", left=True, labelleft=True)
            else:

                ax.tick_params(axis="y", left=False, labelleft=False)

            local_max = max(max(row) for row in percentages) if percentages else 0.0
            threshold = local_max * 0.5
            for row_index, row in enumerate(percentages):
                for col_index, value in enumerate(row):
                    text_color = "white" if value > threshold and value > 0 else "black"
                    ax.text(
                        col_index,
                        row_index,
                        f"{value:.1f}",
                        ha="center",
                        va="center",
                        color=text_color,
                    )

        fig.supxlabel("After propagation")

        if image is not None:
            colorbar = fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02, shrink=0.92)
            colorbar.set_label("Relations (%)", fontsize=10)
            colorbar.ax.tick_params(labelsize=9)

        combined_output_path = iswc_root / "status_transition_heatmaps_iswc.png"
        plt.savefig(combined_output_path, dpi=300)
        plt.close(fig)

    for dataset in ordered_datasets:
        summary[dataset]["output_path"] = str(combined_output_path)
        print(
            f"Status transition subplot included for {dataset} "
            f"(matched={summary[dataset]['matched_relations']}, "
            f"skipped={summary[dataset]['skipped_relations']})"
        )
    print(f"Combined status transition heatmaps saved to {combined_output_path}")

    return summary


def plot_iswc_after_oracle_positive_domain_proportions() -> dict[str, dict[str, float]]:
    """
    Read JSON files in Results/ISWC and compute, for each KG, the proportion
    (0..100) of after_oracle domains having at least one support > 0.

    A common plot for all KGs is saved in Results/ISWC.
    """
    iswc_root = results_root / "ISWC"
    if not iswc_root.exists():
        raise FileNotFoundError(f"Missing ISWC results directory: {iswc_root}")

    json_files = sorted(iswc_root.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under: {iswc_root}")

    per_kg_counts: dict[str, dict[str, int]] = {}

    for json_path in json_files:
        rel_path = json_path.relative_to(iswc_root)
        kg_name = rel_path.parts[0] if len(rel_path.parts) >= 2 else json_path.stem

        with json_path.open("r", encoding="UTF-8") as handle:
            payload = json.load(handle)

        after_oracle = payload.get("after_oracle")
        if not isinstance(after_oracle, Mapping):
            continue

        if kg_name not in per_kg_counts:
            per_kg_counts[kg_name] = {
                "domains_total": 0,
                "domains_with_positive": 0,
                "json_files_seen": 0,
            }

        per_kg_counts[kg_name]["json_files_seen"] += 1

        for raw_domain in after_oracle.values():
            per_kg_counts[kg_name]["domains_total"] += 1
            if _domain_has_positive_support(raw_domain):
                per_kg_counts[kg_name]["domains_with_positive"] += 1

    if not per_kg_counts:
        raise ValueError("No valid after_oracle sections found in ISWC JSON files")

    kg_names = sorted(per_kg_counts.keys())
    percentages: list[float] = []
    summary: dict[str, dict[str, float]] = {}

    for kg_name in kg_names:
        total = per_kg_counts[kg_name]["domains_total"]
        positive = per_kg_counts[kg_name]["domains_with_positive"]
        pct = (positive / total * 100.0) if total > 0 else 0.0
        percentages.append(pct)
        summary[kg_name] = {
            "domains_total": float(total),
            "domains_with_positive": float(positive),
            "percentage_domains_with_positive": pct,
            "json_files_seen": float(per_kg_counts[kg_name]["json_files_seen"]),
        }

    colors = ["#2E86AB", "#F18F01", "#5A9367", "#C73E1D", "#6C5B7B"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(
        kg_names,
        percentages,
        color=[colors[i % len(colors)] for i in range(len(kg_names))],
    )
    ax.set_ylim(0.0, 100.0)
    ax.set_ylabel("Proportion of domains with >=1 support > 0 (%)")
    ax.set_xlabel("TKG")
    ax.set_title("After Oracle - Positive-support domain proportion")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, pct in zip(bars, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{pct:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    output_path = iswc_root / "after_oracle_positive_domain_proportions.png"
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Common histogram saved to {output_path}")
    for kg_name in kg_names:
        pct = summary[kg_name]["percentage_domains_with_positive"]
        positive = int(summary[kg_name]["domains_with_positive"])
        total = int(summary[kg_name]["domains_total"])
        print(f"{kg_name}: {positive}/{total} domains ({pct:.2f}%)")

    return summary


def plot_iswc_support_distributions(bin_width: float = 0.10) -> dict[str, dict[str, Any]]:
    """
    Read all JSON files under Results/ISWC and compute support distributions for
    QCN sections after_oracle and after_propagation, with fixed bins from 0 to 1.

    One histogram figure is generated per KG under Results/ISWC/<KG>/.
    Returns raw counts per KG and per section.
    """
    if bin_width <= 0 or bin_width > 1:
        raise ValueError(f"bin_width must be in (0, 1], got {bin_width}")

    iswc_root = results_root / "ISWC"
    if not iswc_root.exists():
        raise FileNotFoundError(f"Missing ISWC results directory: {iswc_root}")

    json_files = sorted(iswc_root.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under: {iswc_root}")

    label_by_kg = {
        "Q6256": "Country",
        "Q215380": "Musical group",
        "Q82955": "Politician",
    }

    supports_per_kg: dict[str, dict[str, list[float]]] = {}
    for json_path in json_files:
        rel_path = json_path.relative_to(iswc_root)
        kg_name = rel_path.parts[0] if len(rel_path.parts) >= 2 else json_path.stem

        if kg_name not in supports_per_kg:
            supports_per_kg[kg_name] = {
                "after_oracle": [],
                "after_propagation": [],
                "inferred_only": [],
            }

        with json_path.open("r", encoding="UTF-8") as handle:
            payload = json.load(handle)


        for section_name in ("after_oracle", "after_propagation"):
            section = payload.get(section_name)
            if not isinstance(section, Mapping):
                continue
            for raw_domain in section.values():
                supports_per_kg[kg_name][section_name].extend(
                    _extract_supports_from_domain(raw_domain)
                )


        after_oracle = payload.get("after_oracle")
        after_propagation = payload.get("after_propagation")
        if isinstance(after_oracle, Mapping) and isinstance(after_propagation, Mapping):
            for pair_key, propagation_domain in after_propagation.items():
                oracle_domain = after_oracle.get(pair_key, {})
                inferred_only_supports = _extract_inferred_only_supports(oracle_domain, propagation_domain)
                supports_per_kg[kg_name]["inferred_only"].extend(inferred_only_supports)

    summary: dict[str, dict[str, Any]] = {}
    for kg_name, section_values in sorted(supports_per_kg.items()):
        oracle_values = section_values["after_oracle"]
        propagation_values = section_values["after_propagation"]
        inferred_only_values = section_values["inferred_only"]

        oracle_zero_count = sum(1 for value in oracle_values if value == 0.0)
        propagation_zero_count = sum(1 for value in propagation_values if value == 0.0)

        edges, oracle_counts = _build_histogram_counts(oracle_values, bin_width)
        _, propagation_counts = _build_histogram_counts(propagation_values, bin_width)
        _, inferred_only_counts = _build_histogram_counts(inferred_only_values, bin_width)

        x_positions = edges[:-1]
        series_width = bin_width * 0.42

        fig, ax = plt.subplots(1, 1, figsize=(11, 5.5))
        oracle_x_positions = [x + (bin_width * 0.04) for x in x_positions]
        propagation_x_positions = [x + (bin_width * 0.54) for x in x_positions]
        ax.bar(
            oracle_x_positions,
            oracle_counts,
            width=series_width,
            align="edge",
            color="#2E86AB",
            label=f"after oracle",
        )
        ax.bar(
            propagation_x_positions,
            propagation_counts,
            width=series_width,
            align="edge",
            color="#F18F01",
            label=f"after Propagate&Filter",
        )


        if oracle_counts and oracle_zero_count > 0:
            ax.bar(
                [oracle_x_positions[0]],
                [oracle_zero_count],
                width=series_width,
                align="edge",
                facecolor="none",
                edgecolor="black",
                hatch="///",
                linewidth=0.0,
            )
        if propagation_counts and propagation_zero_count > 0:
            ax.bar(
                [propagation_x_positions[0]],
                [propagation_zero_count],
                width=series_width,
                align="edge",
                facecolor="none",
                edgecolor="black",
                hatch="///",
                linewidth=0.0,
            )


        if inferred_only_counts:
            ax.bar(
                propagation_x_positions,
                inferred_only_counts,
                width=series_width,
                align="edge",
                facecolor="none",
                edgecolor="green",
                hatch="\\\\\\\\",
                linewidth=1.0,
            )

        kg_label = label_by_kg.get(kg_name, kg_name)
        ax.set_title(f"{kg_label} ({kg_name}) - Confidence score distribution")
        ax.set_ylabel("Count")
        ax.set_xlabel("Confidence score bin")
        ax.set_yscale("log")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()
        handles.append(
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch="///",
                label=" proportion of 0 confidence in [0, 0.1]",
            )
        )
        labels.append("proportion of 0 confidence in [0, 0.1]")
        handles.append(
            Patch(
                facecolor="white",
                edgecolor="green",
                hatch="\\\\\\\\",
                label=" inferred-only relations (not in oracle)",
            )
        )
        labels.append("inferred-only relations (not in oracle)")
        ax.legend(handles, labels)

        tick_positions = [round(i * bin_width, 10) for i in range(int(round(1.0 / bin_width)) + 1)]
        ax.set_xticks(tick_positions)
        ax.set_xlim(0.0, 1.0)

        plt.tight_layout()
        output_path = iswc_root / kg_name / f"support_histograms_{kg_name}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)

        print(f"Support histograms saved to {output_path}")

        summary[kg_name] = {
            "bins": edges,
            "after_oracle_counts": oracle_counts,
            "after_propagation_counts": propagation_counts,
            "inferred_only_counts": inferred_only_counts,
            "after_oracle_total": len(oracle_values),
            "after_propagation_total": len(propagation_values),
            "inferred_only_total": len(inferred_only_values),
            "after_oracle_zero_count": oracle_zero_count,
            "after_propagation_zero_count": propagation_zero_count,
            "json_files_seen": len(
                [
                    path
                    for path in json_files
                    if (path.relative_to(iswc_root).parts[0] if len(path.relative_to(iswc_root).parts) >= 2 else path.stem) == kg_name
                ]
            ),
        }

    return summary


def plot_qcn3_support_distributions(
    kg_names: list[str] | None = None,
    bin_width: float = 0.10,
) -> dict[str, dict[str, Any]]:
    """
    Read Results/<KG>/qcn3_<KG>.json for each KG and compute support distributions
    for after_oracle and after_propagation.
    One histogram figure (2 axes side by side) is saved to
    Results/<KG>/qcn3_support_histograms_<KG>.png.

    :param kg_names: list of KG names (e.g. ["Q6256", "Q215380"]). If None, auto-discover.
    :param bin_width: histogram bin width on [0, 1].
    """
    if bin_width <= 0 or bin_width > 1:
        raise ValueError(f"bin_width must be in (0, 1], got {bin_width}")

    if kg_names is None:
        kg_names = [
            d.name
            for d in results_root.iterdir()
            if d.is_dir() and (d / f"qcn3_{d.name}.json").exists()
        ]
    if not kg_names:
        raise FileNotFoundError(f"No qcn3_<KG>.json files found under {results_root}")

    summary: dict[str, dict[str, Any]] = {}
    for kg_name in sorted(kg_names):
        json_path = results_root / kg_name / f"qcn3_{kg_name}.json"
        if not json_path.exists():
            print(f"⚠️ Fichier introuvable : {json_path}")
            continue

        with json_path.open("r", encoding="UTF-8") as handle:
            payload = json.load(handle)

        oracle_values: list[float] = []
        propagation_values: list[float] = []
        inferred_only_values: list[float] = []

        for section_name, values_list in (("after_oracle", oracle_values), ("after_propagation", propagation_values)):
            section = payload.get(section_name)
            if not isinstance(section, Mapping):
                continue
            for raw_domain in section.values():
                values_list.extend(_extract_supports_from_domain(raw_domain))

        after_oracle = payload.get("after_oracle")
        after_propagation = payload.get("after_propagation")
        if isinstance(after_oracle, Mapping) and isinstance(after_propagation, Mapping):
            for pair_key, propagation_domain in after_propagation.items():
                oracle_domain = after_oracle.get(pair_key, {})
                inferred_only_values.extend(
                    _extract_inferred_only_supports(oracle_domain, propagation_domain)
                )

        oracle_zero_count = sum(1 for v in oracle_values if v == 0.0)
        propagation_zero_count = sum(1 for v in propagation_values if v == 0.0)
        inferred_zero_count = sum(1 for v in inferred_only_values if v == 0.0)

        edges, oracle_counts = _build_histogram_counts(oracle_values, bin_width)
        _, propagation_counts = _build_histogram_counts(propagation_values, bin_width)
        _, inferred_only_counts = _build_histogram_counts(inferred_only_values, bin_width)

        x_positions = edges[:-1]
        series_width = bin_width * 0.42
        tick_positions = [round(i * bin_width, 10) for i in range(int(round(1.0 / bin_width)) + 1)]

        fig, ax = plt.subplots(1, 1, figsize=(11, 5.5))
        oracle_x = [x + bin_width * 0.04 for x in x_positions]
        propagation_x = [x + bin_width * 0.54 for x in x_positions]

        ax.bar(oracle_x, oracle_counts, width=series_width, align="edge", color="#2E86AB", label="after oracle")
        ax.bar(propagation_x, propagation_counts, width=series_width, align="edge", color="#F18F01", label="after Propagate&Filter")
        if oracle_counts and oracle_zero_count > 0:
            ax.bar([oracle_x[0]], [oracle_zero_count], width=series_width, align="edge",
                   facecolor="none", edgecolor="black", hatch="///", linewidth=0.0)
        if propagation_counts and propagation_zero_count > 0:
            ax.bar([propagation_x[0]], [propagation_zero_count], width=series_width, align="edge",
                   facecolor="none", edgecolor="black", hatch="///", linewidth=0.0)


        if inferred_only_counts:
            ax.bar(
                propagation_x,
                inferred_only_counts,
                width=series_width,
                align="edge",
                facecolor="none",
                edgecolor="green",
                hatch="\\\\\\\\",
                linewidth=1.0,
            )

        ax.set_title(f"{kg_name} (qcn3) - Confidence score distribution")
        ax.set_ylabel("Count")
        ax.set_xlabel("Confidence score bin")
        ax.set_yscale("log")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        handles, labels = ax.get_legend_handles_labels()
        handles.append(
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch="///",
                label="proportion of 0 confidence in [0, 0.1]",
            )
        )
        labels.append("proportion of 0 confidence in [0, 0.1]")
        handles.append(
            Patch(
                facecolor="white",
                edgecolor="green",
                hatch="\\\\\\\\",
                label="inferred-only relations (not in oracle)",
            )
        )
        labels.append("inferred-only relations (not in oracle)")
        ax.legend(handles, labels)

        ax.set_xticks(tick_positions)
        ax.set_xlim(0.0, 1.0)

        plt.tight_layout()
        output_path = results_root / kg_name / f"qcn3_support_histograms_{kg_name}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Support histograms saved to {output_path}")

        summary[kg_name] = {
            "bins": edges,
            "after_oracle_counts": oracle_counts,
            "after_propagation_counts": propagation_counts,
            "inferred_only_counts": inferred_only_counts,
            "after_oracle_total": len(oracle_values),
            "after_propagation_total": len(propagation_values),
            "inferred_only_total": len(inferred_only_values),
            "after_oracle_zero_count": oracle_zero_count,
            "after_propagation_zero_count": propagation_zero_count,
        }

    return summary


def plot_qcn3_domain_size_histograms(
    kg_names: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Plot domain-size histograms for qcn3 results.

        Domain size is the number of relations in a pair-domain:
        - observed size: count of relations with status in
            {Observed, Inferred, Observed and inferred}; only domains with size > 0
            are kept in the observed histogram
    - inferred-only size: count of relations with status Inferred in after_propagation
      that are absent (or status None) in after_oracle.

    Saves one figure per KG to:
    Results/<KG>/qcn3_domain_size_histograms_<KG>.png
    """
    observed_statuses = {"Observed", "Inferred", "Observed and inferred"}

    if kg_names is None:
        kg_names = [
            d.name
            for d in results_root.iterdir()
            if d.is_dir() and (d / f"qcn3_{d.name}.json").exists()
        ]
    if not kg_names:
        raise FileNotFoundError(f"No qcn3_<KG>.json files found under {results_root}")

    summary: dict[str, dict[str, Any]] = {}

    for kg_name in sorted(kg_names):
        json_path = results_root / kg_name / f"qcn3_{kg_name}.json"
        if not json_path.exists():
            print(f"Warning: missing file {json_path}")
            continue

        with json_path.open("r", encoding="UTF-8") as handle:
            payload = json.load(handle)

        after_oracle = payload.get("after_oracle", {})
        after_propagation = payload.get("after_propagation", {})
        if not isinstance(after_oracle, Mapping) or not isinstance(after_propagation, Mapping):
            print(f"Warning: invalid qcn3 payload structure in {json_path}")
            continue

        observed_sizes: list[int] = []
        inferred_only_sizes: list[int] = []
        observed_zero_excluded = 0

        pair_keys = set(after_oracle.keys()) | set(after_propagation.keys())
        for pair_key in pair_keys:
            oracle_domain = after_oracle.get(pair_key, {})
            propagation_domain = after_propagation.get(pair_key, {})

            if not isinstance(oracle_domain, Mapping) or not isinstance(propagation_domain, Mapping):
                continue

            oracle_relations = oracle_domain.get("relations", {})
            propagation_relations = propagation_domain.get("relations", {})
            if not isinstance(oracle_relations, Mapping) or not isinstance(propagation_relations, Mapping):
                continue

            observed_count = 0
            inferred_only_count = 0

            for relation, p_payload in propagation_relations.items():
                if not isinstance(p_payload, Mapping):
                    continue

                p_status = p_payload.get("status")
                if p_status in observed_statuses:
                    observed_count += 1

                o_payload = oracle_relations.get(relation)
                o_status = o_payload.get("status") if isinstance(o_payload, Mapping) else None
                if p_status == "Inferred" and (o_payload is None or o_status is None):
                    inferred_only_count += 1

            if observed_count > 0:
                observed_sizes.append(observed_count)
            else:
                observed_zero_excluded += 1
            inferred_only_sizes.append(inferred_only_count)

        max_size = 0
        if observed_sizes:
            max_size = max(max_size, max(observed_sizes))
        if inferred_only_sizes:
            max_size = max(max_size, max(inferred_only_sizes))
        x_sizes = list(range(max_size + 1))

        observed_hist = [0] * (max_size + 1)
        inferred_only_hist = [0] * (max_size + 1)
        for size in observed_sizes:
            observed_hist[size] += 1
        for size in inferred_only_sizes:
            inferred_only_hist[size] += 1

        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 5.5))

        ax_left.bar(x_sizes, observed_hist, color="#2E86AB", width=0.85)
        ax_left.set_title(f"{kg_name} - Domain size (Observed/Inferred, non-zero)")
        ax_left.set_xlabel("Domain size (#relations)")
        ax_left.set_ylabel("Number of domains")
        ax_left.set_xticks(x_sizes)
        ax_left.grid(axis="y", linestyle="--", alpha=0.5)

        ax_right.bar(x_sizes, inferred_only_hist, color="#2CA02C", width=0.85)
        ax_right.set_title(f"{kg_name} - Domain size (Inferred-only)")
        ax_right.set_xlabel("Domain size (#relations)")
        ax_right.set_ylabel("Number of domains")
        ax_right.set_xticks(x_sizes)
        ax_right.grid(axis="y", linestyle="--", alpha=0.5)

        plt.suptitle(f"{kg_name} (qcn3) - Domain size distributions", fontsize=13, fontweight="bold")
        plt.tight_layout()

        output_path = results_root / kg_name / f"qcn3_domain_size_histograms_{kg_name}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Domain-size histograms saved to {output_path}")

        summary[kg_name] = {
            "sizes": x_sizes,
            "observed_hist": observed_hist,
            "inferred_only_hist": inferred_only_hist,
            "observed_count": len(observed_sizes),
            "observed_zero_excluded": observed_zero_excluded,
            "inferred_only_count": len(inferred_only_sizes),
            "observed_min": min(observed_sizes) if observed_sizes else 0,
            "observed_max": max(observed_sizes) if observed_sizes else 0,
            "observed_avg": (sum(observed_sizes) / len(observed_sizes)) if observed_sizes else 0.0,
            "inferred_only_min": min(inferred_only_sizes) if inferred_only_sizes else 0,
            "inferred_only_max": max(inferred_only_sizes) if inferred_only_sizes else 0,
            "inferred_only_avg": (sum(inferred_only_sizes) / len(inferred_only_sizes)) if inferred_only_sizes else 0.0,
        }

    return summary


def plot_qcn3_after_propagation_domain_size_histogram(
    kg_names: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Plot one histogram per KG for after_propagation domain sizes.

    Domain size = total number of relations present in each domain
    of the after_propagation section.

    Saves one figure per KG to:
    Results/<KG>/qcn3_after_propagation_domain_size_histogram_<KG>.png
    """
    if kg_names is None:
        kg_names = [
            d.name
            for d in results_root.iterdir()
            if d.is_dir() and (d / f"qcn3_{d.name}.json").exists()
        ]
    if not kg_names:
        raise FileNotFoundError(f"No qcn3_<KG>.json files found under {results_root}")

    summary: dict[str, dict[str, Any]] = {}

    for kg_name in sorted(kg_names):
        json_path = results_root / kg_name / f"qcn3_{kg_name}.json"
        if not json_path.exists():
            print(f"Warning: missing file {json_path}")
            continue

        with json_path.open("r", encoding="UTF-8") as handle:
            payload = json.load(handle)

        after_propagation = payload.get("after_propagation", {})
        if not isinstance(after_propagation, Mapping):
            print(f"Warning: invalid after_propagation structure in {json_path}")
            continue

        domain_sizes: list[int] = []
        for domain in after_propagation.values():
            if not isinstance(domain, Mapping):
                continue
            relations = domain.get("relations", {})
            if isinstance(relations, Mapping):
                domain_sizes.append(len(relations))

        if not domain_sizes:
            print(f"Warning: no valid domains found in {json_path}")
            continue

        min_size = min(domain_sizes)
        max_size = max(domain_sizes)
        avg_size = sum(domain_sizes) / len(domain_sizes)

        x_sizes = list(range(min_size, max_size + 1))
        counts = [0] * len(x_sizes)
        for size in domain_sizes:
            counts[size - min_size] += 1

        fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
        ax.bar(x_sizes, counts, color="#1F77B4", width=0.85)
        ax.set_title(f"{kg_name} (qcn3) - After propagation domain sizes")
        ax.set_xlabel("Domain size (#relations)")
        ax.set_ylabel("Number of domains")
        ax.set_xticks(x_sizes)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        subtitle = f"min={min_size}, max={max_size}, avg={avg_size:.2f}, n={len(domain_sizes)}"
        ax.text(0.01, 0.98, subtitle, transform=ax.transAxes, va="top", ha="left")

        plt.tight_layout()
        output_path = results_root / kg_name / f"qcn3_after_propagation_domain_size_histogram_{kg_name}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"After-propagation domain-size histogram saved to {output_path}")

        summary[kg_name] = {
            "sizes": x_sizes,
            "counts": counts,
            "num_domains": len(domain_sizes),
            "min_size": min_size,
            "max_size": max_size,
            "avg_size": avg_size,
        }

    return summary

def plot_iswc_after_propagation_inferred_support_histogram(
    bin_width: float = 0.10,
) -> dict[str, Any]:
    """
    Build one global histogram of supports for relations in after_propagation
    whose normalized status is exactly "inferred".

    The figure is saved under Results/ISWC.
    """
    if bin_width <= 0 or bin_width > 1:
        raise ValueError(f"bin_width must be in (0, 1], got {bin_width}")

    iswc_root = results_root / "ISWC"
    if not iswc_root.exists():
        raise FileNotFoundError(f"Missing ISWC results directory: {iswc_root}")

    json_files = sorted(iswc_root.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under: {iswc_root}")

    supports: list[float] = []
    inferred_relations_per_kg: dict[str, int] = {}

    for json_path in json_files:
        rel_path = json_path.relative_to(iswc_root)
        kg_name = rel_path.parts[0] if len(rel_path.parts) >= 2 else json_path.stem

        with json_path.open("r", encoding="UTF-8") as handle:
            payload = json.load(handle)

        after_propagation = payload.get("after_propagation")
        if not isinstance(after_propagation, Mapping):
            continue

        inferred_relations_per_kg.setdefault(kg_name, 0)

        for raw_domain in after_propagation.values():
            if not isinstance(raw_domain, Mapping):
                continue

            structured_relations = raw_domain.get("relations")
            if not isinstance(structured_relations, Mapping):
                continue

            for relation_payload in structured_relations.values():
                if not isinstance(relation_payload, Mapping):
                    continue

                status = _normalize_transition_status(relation_payload.get("status"))
                if status != "inferred":
                    continue

                support = relation_payload.get("support")
                if support is None or not isinstance(support, (int, float)):
                    continue
                support_value = float(support)
                if 0.0 <= support_value <= 1.0:
                    supports.append(support_value)
                    inferred_relations_per_kg[kg_name] += 1

    if not supports:
        raise ValueError(
            "No after_propagation relation with status 'inferred' and valid support in [0, 1]"
        )

    edges, counts = _build_histogram_counts(supports, bin_width)
    x_positions = edges[:-1]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(
        x_positions,
        counts,
        width=bin_width * 0.9,
        align="edge",
        color="#F18F01",
        edgecolor="black",
        linewidth=0.4,
    )

    tick_positions = [
        round(i * bin_width, 10)
        for i in range(int(round(1.0 / bin_width)) + 1)
    ]
    ax.set_xticks(tick_positions)
    ax.set_xlim(0.0, 1.0)
    ax.set_yscale("log")
    ax.set_xlabel("Confidence score bin")
    ax.set_ylabel("Count of inferred relations")
    ax.set_title("After propagation - support distribution for inferred relations")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path = iswc_root / "after_propagation_inferred_support_histogram.png"
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Inferred-support histogram saved to {output_path}")

    return {
        "bins": edges,
        "counts": counts,
        "total_inferred_relations": len(supports),
        "inferred_relations_per_kg": inferred_relations_per_kg,
        "json_files_seen": len(json_files),
        "output_path": str(output_path),
    }


def checkConsistency(jsonfile: str, threshold: float = 1.0) -> bool:
    '''
    Check the consistency of qcn loaded from a json file
    '''
    def to_structured_qcn(
        legacy_qcn: dict[tuple[str, str], dict[str, float]]
    ) -> dict[tuple[str, str], dict[str, object]]:
        structured_qcn: dict[tuple[str, str], dict[str, object]] = {}
        for pair, domain in legacy_qcn.items():
            structured_qcn[pair] = {
                "relations": {
                    relation: {"support": float(score), "status": None}
                    for relation, score in domain.items()
                }
            }
        return structured_qcn

    qcn = read_qcn_from_json(jsonfile)



    qcn = _prepare_qcn_for_threshold(qcn, threshold)

    structured_qcn = to_structured_qcn(qcn)

    return path_consistency_classique(structured_qcn) is not None






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check consistency of a QCN from a JSON file.")
    parser.add_argument("jsonfile", type=str, help="Path to the QCN JSON file.")
    parser.add_argument("--threshold", type=float, default=1.0, help="Score threshold for consistency check.")
    parser.add_argument(
        "--incoherence-report",
        action="store_true",
        help="Generate a detailed report with contradictory triplets and incoherence path.",
    )
    parser.add_argument(
        "--max-triplets",
        type=int,
        default=300,
        help="Maximum number of contradictory triplets written in the report.",
    )
    parser.add_argument(
        "--trace-tail",
        type=int,
        default=30,
        help="Number of propagation steps kept before the collapse event.",
    )
    args = parser.parse_args()

    if args.incoherence_report:
        report = incoherence_report(
            args.jsonfile,
            threshold=args.threshold,
            max_triplets=args.max_triplets,
            trace_tail=args.trace_tail,
        )
        print(f"Incoherence report written to: {report}")

    is_consistent = checkConsistency(args.jsonfile, threshold=args.threshold)
    if is_consistent:
        print("The QCN is consistent.")
    else:
        print("The QCN is inconsistent.")

