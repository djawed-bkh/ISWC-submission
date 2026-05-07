import argparse
import copy
import json
from collections.abc import Mapping
from collections import deque
import os
from itertools import combinations
import multiprocessing as mp
import time

from . import allen_relations as AllenR
from .allen_list import ALLEN_RELATIONS
from .generator_common import (
    RESULTS_ROOT as results_root,
    build_scored_networkwithThreshold as build_entity_network,
    compose_allen_scores as composition_Allen,
    converse_domain_scores as converse_domain,
    read_kg as read_KG,
)


PROPERTY_SUPPORT_THRESHOLD = 0.4  # Garde les 40% de propriétés les plus présentes en quadruplets
SUPPORT_KEY = "support"
STATUS_KEY = "status"
RELATIONS_KEY = "relations"
STATUS_OBSERVED = "Observed"
STATUS_OBSERVED_AND_INFERRED = "Observed and inferred"
STATUS_REPAIRED = "Repaired"
STATUS_INFERRED = "Inferred"
STATUS_INFERRED_LEGACY_TYPO = "Infered"


def normalize_relation_status(status: object) -> str | None:
    """Normalize status labels to the canonical set used by learner2."""
    if not isinstance(status, str):
        return None

    normalized = status.strip().lower()
    if normalized == "observed":
        return STATUS_OBSERVED
    if normalized in {"observed and inferred", "inferred and observed"}:
        return STATUS_OBSERVED_AND_INFERRED
    if normalized in {"inferred", "infered"}:
        return STATUS_INFERRED
    if normalized == "repaired":
        return STATUS_REPAIRED
    return status


def relation_status(observed: bool, inferred: bool) -> str | None:
    """Return the relation status from observed/inferred booleans."""
    if observed and inferred:
        return STATUS_OBSERVED_AND_INFERRED
    if observed:
        return STATUS_OBSERVED
    if inferred:
        return STATUS_INFERRED
    return None


def relation_entries(domain: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    """Extract per-relation payloads from a constraint domain."""
    relations_obj = domain.get(RELATIONS_KEY)
    if not isinstance(relations_obj, Mapping):
        raise ValueError(
            "Invalid scored domain format: expected a 'relations' mapping for each constraint domain."
        )
    entries: dict[str, Mapping[str, object]] = {}
    for relation, payload in relations_obj.items():
        if relation not in ALLEN_RELATIONS:
            continue
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"Invalid scored domain format: relation '{relation}' must map to a payload."
            )
        if SUPPORT_KEY not in payload or STATUS_KEY not in payload:
            raise ValueError(
                f"Invalid scored domain format: relation '{relation}' must contain '{SUPPORT_KEY}' and '{STATUS_KEY}'."
            )
        support = payload.get(SUPPORT_KEY)
        status = payload.get(STATUS_KEY)
        if support is not None and not isinstance(support, (int, float)):
            raise ValueError(
                f"Invalid scored domain format: relation '{relation}' has a non numeric support."
            )
        if status is not None and not isinstance(status, str):
            raise ValueError(
                f"Invalid scored domain format: relation '{relation}' has a non string status."
            )
        entries[relation] = payload
    return entries


def relation_scores(domain: Mapping[str, object]) -> dict[str, float]:
    """Extract numeric supports from a constraint domain."""
    scores: dict[str, float] = {}
    for relation, payload in relation_entries(domain).items():
        support = payload.get(SUPPORT_KEY)
        if isinstance(support, (int, float)):
            scores[relation] = float(support)
    return scores


def relation_statuses(domain: Mapping[str, object]) -> dict[str, str | None]:
    """Extract relation statuses from a constraint domain."""
    statuses: dict[str, str | None] = {}
    for relation, payload in relation_entries(domain).items():
        status = payload.get(STATUS_KEY)
        statuses[relation] = normalize_relation_status(status)
    return statuses


def with_relation_data(
    relations: Mapping[str, float | None],
    statuses: Mapping[str, str | None] | None = None,
) -> dict[str, object]:
    """Build a domain with support/status stored per relation."""
    relation_status_map = statuses or {}
    return {
        RELATIONS_KEY: {
            relation: {
                SUPPORT_KEY: support,
                STATUS_KEY: relation_status_map.get(relation),
            }
            for relation, support in relations.items()
        }
    }


def build_output_stem(kg_type: str) -> str:
    """Build the base filename stem for Generator2 outputs."""
    return f"qcn2_{kg_type}"


        # -2 : score négatif spécial pour indiquer que aucune entité n'a les deux propriétés, ce qui signifie que le KG ne fournit aucune information pour ce couple de propriétés, et que l'on ne peut pas conclure à l'existence ou l'inexistence de la relation d'Allen entre ces propriétés à partir du KG
        # -1 : score négatif spécial hérité d'anciennes sorties, représentant une relation non fiable qui ne doit pas être propagée dans les compositions


def heuristic_queue(
    qcn: dict[tuple[str, str], dict[str, object]], type: str = "random"
) -> deque[tuple[str, str]]:
    """
    Réordonne la file de propagation en donnant la priorité aux paires avec les domaines les plus restreints afin de maximiser l'efficacité de la propagation.
    """
    queue = deque(qcn.keys())  # initialisation de la file avec toutes les paires
    # Calculer la taille des domaines pour chaque paire dans la file
    pair_domain_sizes = {pair: len(relation_entries(qcn[pair])) for pair in queue}

    # Trier les paires par taille de domaine (du plus petit au plus grand)
    if type == "domain_size_asc":
        sorted_pairs = sorted(pair_domain_sizes, key=lambda pair: pair_domain_sizes[pair])
    elif type == "domain_size_desc":
        sorted_pairs = sorted(
            pair_domain_sizes,
            key=lambda pair: pair_domain_sizes[pair],
            reverse=True,
        )
    elif type == "random":
        import random
        sorted_pairs = list(pair_domain_sizes.keys())
        random.shuffle(sorted_pairs)
    else:
        raise ValueError(f"Unknown heuristic type: {type}")
    # Réordonner la file en fonction de ce tri
    queue.clear()
    queue.extend(sorted_pairs)
    return queue

def propagateAndFilter(qcn: dict) -> dict:
    '''
    prend en entrée le qcn résultant du learner. 
    '''
    qcn_copy = copy.deepcopy(qcn)
    composition_source_statuses = {
        STATUS_OBSERVED,
        STATUS_OBSERVED_AND_INFERRED,
    }

    def composable_relations(domain: Mapping[str, object]) -> dict[str, float]:
        """Return relations eligible to contribute to composition."""
        scores = relation_scores(domain)
        statuses = relation_statuses(domain)
        return {
            relation: score
            for relation, score in scores.items()
            if score > 0 and statuses.get(relation) in composition_source_statuses
        }

    def domain_has_observed_status(domain: Mapping[str, object]) -> bool:
        """Return True if a domain contains observed evidence."""
        statuses = relation_statuses(domain)
        return any(
            status in composition_source_statuses
            for status in statuses.values()
        )

    properties = sorted(
        {p for (x, y) in qcn_copy.keys() for p in (x, y)}
    )  # liste des propriétés impliquées dans le réseau de contraintes

    queue = deque(qcn_copy.keys())  # initialisation de la file avec toutes les paires

    while queue:  # tant que la file n'est pas vide
        i, j = queue.popleft()  # dépiler une paire (i,j)

        for k in properties:
            if k == i or k == j:
                continue
            if (i, k) not in qcn_copy or (k, j) not in qcn_copy:
                print(
                    "Warning: missing arc in qcn_copy during propagation:",
                    (i, k),
                    "or",
                    (k, j),
                    "this should not happen !",
                )
                raise ValueError(
                    f"Missing arc in qcn_copy during propagation: {(i, k)} or {(k, j)}"
                )
            
            # Keep only strictly positive supports for propagation targets.
            ik_relations = {
                relation: score
                for relation, score in relation_scores(qcn_copy[(i, k)]).items()
                if score > 0
            }  # beneficiaire de la composition
            kj_relations = {
                relation: score
                for relation, score in relation_scores(qcn_copy[(k, j)]).items()
                if score > 0
            }  # beneficiaire de la composition
            
            ij_relations = relation_scores(qcn_copy[(i, j)]) # utilisé pour la composition de ik et kj
            jk_relations = relation_scores(qcn_copy[(j, k)]) # utilisé pour la composition de ik
            ki_relations = relation_scores(qcn_copy[(k, i)]) # utilisé pour la composition de ik

            relations_to_compose_ij = composable_relations(qcn_copy[(i, j)])
            relations_to_compose_jk = composable_relations(qcn_copy[(j, k)])
            relations_to_compose_ki = composable_relations(qcn_copy[(k, i)])

            # check si il existe au moins une relation composable (support > 0 + statut autorisé)
            positive_support_ij = bool(relations_to_compose_ij)
            positive_support_jk = bool(relations_to_compose_jk)
            positive_support_ki = bool(relations_to_compose_ki)
            target_ik_is_observed = domain_has_observed_status(qcn_copy[(i, k)])
            target_kj_is_observed = domain_has_observed_status(qcn_copy[(k, j)])

            # Traiter la paire (i, k)
            if (not target_ik_is_observed) and positive_support_ij and positive_support_jk:
                # On fait la composition seulement avec les relations composables.
                composition = composition_Allen(relations_to_compose_ij, relations_to_compose_jk)
                for c in composition:
                    if c not in ik_relations: # si la relation composée n'est pas dans ik_relations
                        # recuperer le statut de c dans ik
                        ik_relation_status = relation_statuses(qcn_copy[(i, k)]).get(c)
                        if ik_relation_status is  not None:
                            print("Warning: we should not be here, if the relation is not in ik_relations, it should not have a status different from None")
                            raise ValueError("Unexpected case in propagation: relation not in ik_relations but has an observed status")                      
                        #update score et statut
                        qcn_copy[(i, k)][RELATIONS_KEY][c] = {SUPPORT_KEY: composition[c], STATUS_KEY: STATUS_INFERRED}
                        qcn_copy[(k, i)] = converse_domain(qcn_copy[(i, k)])
                        if (i, k) not in queue:
                            queue.append((i, k))
                        
                    elif composition[c] > ik_relations[c]:  # si le score de la relation composée est supérieur au score actuel dans ik_relations
                        
                       
                        ik_relation_status = relation_statuses(qcn_copy[(i, k)]).get(c)
                        if ik_relation_status == STATUS_OBSERVED:
                            qcn_copy[(i, k)][RELATIONS_KEY][c] = {
                                SUPPORT_KEY: ik_relations[c],
                                STATUS_KEY: STATUS_OBSERVED_AND_INFERRED,
                            }
                            qcn_copy[(k, i)] = converse_domain(qcn_copy[(i, k)])
                        elif ik_relation_status == STATUS_OBSERVED_AND_INFERRED:
                            # No-op: observed support remains authoritative.
                            pass
                        elif ik_relation_status == STATUS_INFERRED:
                            qcn_copy[(i, k)][RELATIONS_KEY][c] = {SUPPORT_KEY: composition[c], STATUS_KEY: STATUS_INFERRED}                        
                            qcn_copy[(k, i)] = converse_domain(qcn_copy[(i, k)])
                            if (i, k) not in queue:
                                queue.append((i, k))
                        else:
                            print("Warning: we should not be here")
                            raise ValueError("Unexpected case in propagation: relation not in ik_relations but composition score is higher than ik_relations score")
        
                    elif composition[c] <= ik_relations[c]: # si le score de la relation composée est inférieur ou égal au score actuel dans ik_relations,
                        #update score et statut
                        ik_relation_status = relation_statuses(qcn_copy[(i, k)]).get(c)
                        if ik_relation_status == STATUS_OBSERVED:
                            qcn_copy[(i, k)][RELATIONS_KEY][c] = {
                                SUPPORT_KEY: ik_relations[c],
                                STATUS_KEY: STATUS_OBSERVED_AND_INFERRED,
                            }
                            qcn_copy[(k, i)] = converse_domain(qcn_copy[(i, k)])
                        elif ik_relation_status in {STATUS_INFERRED, STATUS_OBSERVED_AND_INFERRED}:
                            # No-op: no improvement and status already carries inference.
                            pass
                        else:
                            print("Warning: we should not be here")
                            raise ValueError("Unexpected case in propagation: relation not in ik_relations but composition score is higher than ik_relations score")


            # Traiter la paire (k,j)
            if (not target_kj_is_observed) and positive_support_ki and positive_support_ij:
                # On fait la composition seulement avec les relations composables.
                composition = composition_Allen(relations_to_compose_ki, relations_to_compose_ij)
                for c in composition:
                    if c not in kj_relations: # si la relation composée n'est pas dans kj_relations
                        # recuperer le statut de c dans kj
                        kj_relation_status = relation_statuses(qcn_copy[(k, j)]).get(c)
                        if kj_relation_status is  not None:
                            print("Warning: we should not be here, if the relation is not in kj_relations, it should not have a status different from None")
                            raise ValueError("Unexpected case in propagation: relation not in kj_relations but has an observed status")                      
                        #update score et statut
                        qcn_copy[(k, j)][RELATIONS_KEY][c] = {SUPPORT_KEY: composition[c], STATUS_KEY: STATUS_INFERRED}
                        qcn_copy[(j, k)] = converse_domain(qcn_copy[(k, j)])
                        if (k, j) not in queue:
                            queue.append((k, j))
                        
                    elif composition[c] > kj_relations[c]:  # si le score de la relation composée est supérieur au score actuel dans kj_relations
                        
                       
                        kj_relation_status = relation_statuses(qcn_copy[(k, j)]).get(c)
                        if kj_relation_status == STATUS_OBSERVED:
                            qcn_copy[(k, j)][RELATIONS_KEY][c] = {
                                SUPPORT_KEY: kj_relations[c],
                                STATUS_KEY: STATUS_OBSERVED_AND_INFERRED,
                            }
                            qcn_copy[(j, k)] = converse_domain(qcn_copy[(k, j)])
                        elif kj_relation_status == STATUS_OBSERVED_AND_INFERRED:
                            # No-op: observed support remains authoritative.
                            pass
                        elif kj_relation_status == STATUS_INFERRED:
                            qcn_copy[(k, j)][RELATIONS_KEY][c] = {SUPPORT_KEY: composition[c], STATUS_KEY: STATUS_INFERRED}                        
                            qcn_copy[(j, k)] = converse_domain(qcn_copy[(k, j)])
                            if (k, j) not in queue:
                                queue.append((k, j))
                        else:
                            print("Warning: we should not be here")
                            raise ValueError("Unexpected case in propagation: relation not in kj_relations but composition score is higher than kj_relations score")
        
                    elif composition[c] <= kj_relations[c]: # si le score de la relation composée est inférieur ou égal au score actuel dans kj_relations,
                        #update score et statut
                        kj_relation_status = relation_statuses(qcn_copy[(k, j)]).get(c)
                        if kj_relation_status == STATUS_OBSERVED:
                            qcn_copy[(k, j)][RELATIONS_KEY][c] = {
                                SUPPORT_KEY: kj_relations[c],
                                STATUS_KEY: STATUS_OBSERVED_AND_INFERRED,
                            }
                            qcn_copy[(j, k)] = converse_domain(qcn_copy[(k, j)])
                        elif kj_relation_status in {STATUS_INFERRED, STATUS_OBSERVED_AND_INFERRED}:
                            # No-op: no improvement and status already carries inference.
                            pass
                        else:
                            print("Warning: we should not be here")
                            raise ValueError("Unexpected case in propagation: relation not in kj_relations but composition score is higher than kj_relations score")

    # Ne conserver que les relations observées et/ou inférées.
    allowed_statuses = {
        STATUS_OBSERVED,
        STATUS_INFERRED,
        STATUS_OBSERVED_AND_INFERRED,
    }
    for pair, domain in qcn_copy.items():       #filtrage
        relations_to_remove = []
        for relation, payload in relation_entries(domain).items():
            status = normalize_relation_status(payload.get(STATUS_KEY))
            if status not in allowed_statuses:
                relations_to_remove.append(relation)

        for relation in relations_to_remove:
            del qcn_copy[pair][RELATIONS_KEY][relation]

    return qcn_copy



def learner2(qcn: dict, entities: dict, properties: dict) -> dict | None:
    """
    Learner threshold-free with an oracle phase followed by propagation/filtering.
    Chaque relation du QCN porte un score correspondant à la proportion d'entités
    vérifiant la relation d'Allen pour le couple de propriétés considéré.
    """

    cpu_start = time.process_time()
    oracle_phase_start = cpu_start
    initial_qcn = copy.deepcopy(qcn)
    kept_properties = set(properties)
    kept_facts_count = 0
    for entity in entities.values():
        for prop, triples in entity.triples_per_p.items():
            if prop in kept_properties:
                kept_facts_count += len(triples)

    total_oracle_calls = 0
    min_oracle_cpu_time = float('inf')
    max_oracle_cpu_time = float('-inf')
    oracle_sum_cpu_time = 0.0
    for p1, p2 in combinations(properties, 2):
        if (p1, p2) not in qcn:
            print(
                "Warning: missing arc in qcn for pair:",
                (p1, p2),
                "this should not happen !",
            )
            raise ValueError(f"Missing arc in qcn for pair: {(p1, p2)}")
        if (p2, p1) not in qcn:
            print(
                "Warning: missing arc in qcn for pair:",
                (p2, p1),
                "this should not happen !",
            )
            raise ValueError(f"Missing arc in qcn for pair: {(p2, p1)}")

        V = (p1, p2)
        oraclestart = time.process_time()

        query_result = Query(V, entities)  # threshold free
        oracle_end_time = time.process_time() - oraclestart
        total_oracle_calls += 1
        min_oracle_cpu_time = min(min_oracle_cpu_time, oracle_end_time)
        max_oracle_cpu_time = max(max_oracle_cpu_time, oracle_end_time)
        oracle_sum_cpu_time += oracle_end_time

        if query_result is not None:
            qcn[(p1, p2)] = copy.deepcopy(query_result)

            # Mettre à jour la paire inverse avec les relations inverses
            qcn[(p2, p1)] = converse_domain(qcn[(p1, p2)])

            if not relation_entries(
                qcn[(p1, p2)]
            ):  # si le domaine de C(p1,p2) est vide, on arrête l'apprentissage (incohérence)
                print(f"COLLAPSE: apres la requête sur {V} ! ORIGINE: ORACLE")
                return None

    oracle_phase_cpu_endtime = time.process_time() - oracle_phase_start
    # Sauvegarder le QCN après oracle
    qcn_after_oracle = copy.deepcopy(qcn)

    # Statistiques sur le QCN après oracle
    qcn_stats = compute_qcn_relation_stats(qcn_after_oracle)
    after_oracle_report_stats = compute_after_oracle_report_stats(qcn_after_oracle)
    avg_oracle_cpu = oracle_sum_cpu_time / total_oracle_calls if total_oracle_calls > 0 else 0.0
    min_oracle_cpu_time_safe = min_oracle_cpu_time if total_oracle_calls > 0 else 0.0
    max_oracle_cpu_time_safe = max_oracle_cpu_time if total_oracle_calls > 0 else 0.0

    print(f"\n{'─' * 60}")
    print("  STATISTIQUES QCN — après phase Oracle")
    print(f"{'─' * 60}")
    print(f"  Domaines (paires)          : {qcn_stats['num_domains']}")
    print(f"  Relations totales          : {qcn_stats['total_relations']}")
    print(
        f"  Rel observed & sup > 0     : {qcn_stats['count_observed']}"
    )
    print(
        f"    -> proportion            : {qcn_stats['count_observed']} / "
        f"{qcn_stats['total_relations']} "
        f"({qcn_stats['percent_observed_sup_pos']:.2f}%)"
    )
    print(
        f"  Rel status=None & sup is None : {qcn_stats['count_none_none']}"
    )
    print(
        f"    -> proportion            : {qcn_stats['count_none_none']} / "
        f"{qcn_stats['total_relations']} "
        f"({qcn_stats['percent_none_none']:.2f}%)"
    )
    print(
        f"  Rel status=None & sup == 0 : {qcn_stats['count_none_zero']}"
    )
    print(
        f"    -> proportion            : {qcn_stats['count_none_zero']} / "
        f"{qcn_stats['total_relations']} "
        f"({qcn_stats['percent_none_zero']:.2f}%)"
    )
    print(
        f"  Rel inferred               : {qcn_stats['count_inferred']}"
    )
    print(
        f"  Rel observed and inferred  : {qcn_stats['count_observed_and_inferred']}"
    )
    print(
        f"  % domaines sans relation observée : "
        f"{qcn_stats['count_domains_without_observed']} / {qcn_stats['num_domains']} "
        f"({qcn_stats['percent_domains_without_observed']:.2f}%)"
    )
    print(
        f"  % domaines avec >=1 rel observée & sup>0 : "
        f"{qcn_stats['count_domains_with_observed_sup_pos']} / {qcn_stats['num_domains']} "
        f"({qcn_stats['percent_domains_with_observed_sup_pos']:.2f}%)"
    )
    print(
        f"  Support > 0 / domaine      : moy={qcn_stats['avg_support_per_domain']:.2f}"
        f"  min={qcn_stats['min_support_per_domain']}"
        f"  max={qcn_stats['max_support_per_domain']}"
        f"  σ={qcn_stats['std_support_per_domain']:.2f}"
    )
    print(f"{'─' * 60}")
    print("  TEMPS CPU — phase Oracle")
    print(f"{'─' * 60}")
    print(f"  Propriétés gardées          : {len(kept_properties)}")
    print(f"  Faits gardés                : {kept_facts_count}")
    print(f"  Appels oracle              : {total_oracle_calls}")
    print(f"  Coût total phase oracle    : {oracle_phase_cpu_endtime:.4f} s")
    print(
        f"  Temps oracle / appel — moy : {avg_oracle_cpu:.6f} s"
        f"  min={min_oracle_cpu_time_safe:.6f} s"
        f"  max={max_oracle_cpu_time_safe:.6f} s"
    )
    print(f"{'─' * 60}\n")

    propagate_and_filter_start = time.process_time()
    qcn = propagateAndFilter(qcn)
    propagate_and_filter_cpu_time = time.process_time() - propagate_and_filter_start

    qcn_after_propagation = copy.deepcopy(qcn)
    qcn_after_propagation_stats = compute_qcn_relation_stats(qcn_after_propagation)
    oracle_to_propagation_transition_stats = (
        compute_oracle_to_propagation_transition_stats(
            initial_qcn,
            qcn_after_oracle,
            qcn_after_propagation,
        )
    )
    learner_total_cpu_time = time.process_time() - cpu_start
    cumulative_after_oracle = oracle_phase_cpu_endtime
    cumulative_after_propagation = (
        oracle_phase_cpu_endtime + propagate_and_filter_cpu_time
    )

    print(f"{'─' * 60}")
    print("  TEMPS CPU — learner2")
    print(f"{'─' * 60}")
    print(f"  Oracle                          : {oracle_phase_cpu_endtime:.4f} s")
    print(f"  Propagation + filtre            : {propagate_and_filter_cpu_time:.4f} s")
    print(f"  Total learner2                  : {learner_total_cpu_time:.4f} s")
    print(f"{'─' * 60}\n")

    return {
        "initial": initial_qcn,
        "after_oracle": qcn_after_oracle,
        "after_propagation": qcn_after_propagation,
        "oracle_stats": {
            "total_calls": total_oracle_calls,
            "min_cpu_time": min_oracle_cpu_time_safe,
            "max_cpu_time": max_oracle_cpu_time_safe,
            "avg_cpu_time": avg_oracle_cpu,
            "cost_of_the_oracle_phase": oracle_phase_cpu_endtime,
        },
        "property_filter_stats": {
            "top_k_fraction": PROPERTY_SUPPORT_THRESHOLD,
            "kept_properties_count": len(kept_properties),
            "kept_facts_count": kept_facts_count,
        },
        "qcn_stats": qcn_stats,
        "after_oracle_report_stats": after_oracle_report_stats,
        "after_propagation_qcn_stats": qcn_after_propagation_stats,
        "oracle_to_propagation_transition_stats": oracle_to_propagation_transition_stats,
        "timing": {
            "oracle_phase_cpu_time": oracle_phase_cpu_endtime,
            "propagate_and_filter_cpu_time": propagate_and_filter_cpu_time,
            "learner_total_cpu_time": learner_total_cpu_time,
            "cumulative_after_oracle_cpu_time": cumulative_after_oracle,
            "cumulative_after_propagation_cpu_time": cumulative_after_propagation,
        },

        #"stats": reduction_stats,
    }


def count_total_relations(qcn):
    """Compte le nombre total de relations d'Allen dans tous les domaines"""
    total = 0
    for domain in qcn.values():
        total += len(relation_entries(domain))
    return total


def compute_qcn_relation_stats(qcn: dict) -> dict:
    """
    Calcule des statistiques sur les relations d'un QCN :
    - nombre total de relations
        - nombre de relations par statut métier (Observed, Observed and inferred,
            Inferred, status=None avec support None/0)
        - pour chaque domaine : nombre de relations avec support > 0
      (min, max, moyenne, écart-type sur l'ensemble des domaines)
    """
    import math

    total_relations = 0
    count_observed = 0
    count_observed_and_inferred = 0
    count_inferred = 0
    count_none_none = 0
    count_none_zero = 0
    count_domains_without_observed = 0
    count_domains_with_observed_sup_pos = 0
    support_counts_per_domain: list[int] = []

    for domain in qcn.values():
        entries = relation_entries(domain)
        total_relations += len(entries)
        domain_support_count = 0
        domain_has_observed = False
        domain_has_observed_sup_pos = False
        for payload in entries.values():
            status = normalize_relation_status(payload.get(STATUS_KEY))
            support = payload.get(SUPPORT_KEY)
            if status == STATUS_OBSERVED:
                count_observed += 1
                domain_has_observed = True
            elif status == STATUS_OBSERVED_AND_INFERRED:
                count_observed_and_inferred += 1
                domain_has_observed = True
            elif status == STATUS_INFERRED:
                count_inferred += 1
            elif status is None and support is None:
                count_none_none += 1
            elif status is None and isinstance(support, (int, float)) and support == 0:
                count_none_zero += 1

            if status in {STATUS_OBSERVED, STATUS_OBSERVED_AND_INFERRED} and isinstance(support, (int, float)) and support > 0:
                domain_has_observed_sup_pos = True
            if isinstance(support, (int, float)) and support > 0:
                domain_support_count += 1

        if not domain_has_observed:
            count_domains_without_observed += 1
        if domain_has_observed_sup_pos:
            count_domains_with_observed_sup_pos += 1
        support_counts_per_domain.append(domain_support_count)

    n = len(support_counts_per_domain)
    if n > 0:
        avg_support = sum(support_counts_per_domain) / n
        variance = sum((x - avg_support) ** 2 for x in support_counts_per_domain) / n
        std_support = math.sqrt(variance)
        min_support = min(support_counts_per_domain)
        max_support = max(support_counts_per_domain)
    else:
        avg_support = std_support = min_support = max_support = 0.0

    percent_observed_sup_pos = (
        ((count_observed + count_observed_and_inferred) / total_relations * 100)
        if total_relations > 0
        else 0.0
    )
    percent_none_none = (
        (count_none_none / total_relations * 100) if total_relations > 0 else 0.0
    )
    percent_none_zero = (
        (count_none_zero / total_relations * 100) if total_relations > 0 else 0.0
    )
    percent_domains_without_observed = (
        (count_domains_without_observed / n * 100) if n > 0 else 0.0
    )
    percent_domains_with_observed_sup_pos = (
        (count_domains_with_observed_sup_pos / n * 100) if n > 0 else 0.0
    )

    return {
        "num_domains": n,
        "total_relations": total_relations,
        "count_observed": count_observed,
        "count_observed_and_inferred": count_observed_and_inferred,
        "count_inferred": count_inferred,
        "count_none_none": count_none_none,
        "count_none_zero": count_none_zero,
        "percent_observed_sup_pos": percent_observed_sup_pos,
        "percent_none_none": percent_none_none,
        "percent_none_zero": percent_none_zero,
        "count_domains_without_observed": count_domains_without_observed,
        "count_domains_with_observed_sup_pos": count_domains_with_observed_sup_pos,
        "percent_domains_without_observed": percent_domains_without_observed,
        "percent_domains_with_observed_sup_pos": percent_domains_with_observed_sup_pos,
        "avg_support_per_domain": avg_support,
        "std_support_per_domain": std_support,
        "min_support_per_domain": min_support,
        "max_support_per_domain": max_support,
    }


def compute_after_oracle_report_stats(qcn: dict) -> dict:
    """Compute after-oracle counters used in the text report."""
    total_domains = len(qcn)
    total_relations = 0
    observed_support_pos_relations = 0
    zero_support_relations = 0
    domains_with_observed_status = 0
    domains_with_none_support = 0

    for domain in qcn.values():
        entries = relation_entries(domain)
        total_relations += len(entries)

        has_observed_status = False
        has_none_support = False

        # Empty domain means no evidence remained for this property pair.
        if not entries:
            has_none_support = True

        for payload in entries.values():
            status = normalize_relation_status(payload.get(STATUS_KEY))
            support = payload.get(SUPPORT_KEY)

            if isinstance(support, (int, float)) and support > 0:
                observed_support_pos_relations += 1
            elif isinstance(support, (int, float)) and support == 0:
                zero_support_relations += 1

            if status == STATUS_OBSERVED:
                has_observed_status = True
            if support is None:
                has_none_support = True

        if has_observed_status:
            domains_with_observed_status += 1
        if has_none_support:
            domains_with_none_support += 1

    return {
        "total_domains": total_domains,
        "total_relations": total_relations,
        "observed_support_pos_relations": observed_support_pos_relations,
        "zero_support_relations": zero_support_relations,
        "domains_with_observed_status": domains_with_observed_status,
        "domains_with_none_support": domains_with_none_support,
    }


def compute_oracle_to_propagation_transition_stats(
    initial_qcn: dict,
    after_oracle_qcn: dict,
    after_propagation_qcn: dict,
) -> dict:
    """Compute transition counters between oracle and propagation stages."""
    none_after_oracle_total = 0
    none_to_inferred_count = 0
    observed_after_oracle_total = 0
    observed_to_observed_and_inferred_count = 0
    domains_without_positive_support_after_oracle = 0
    domains_without_positive_support_to_positive_after_propagation = 0

    for pair, oracle_domain in after_oracle_qcn.items():
        oracle_entries = relation_entries(oracle_domain)
        propagation_entries = relation_entries(after_propagation_qcn.get(pair, {}))

        oracle_has_positive_support = False
        propagation_has_positive_support = False

        for relation, oracle_payload in oracle_entries.items():
            oracle_status = normalize_relation_status(oracle_payload.get(STATUS_KEY))
            oracle_support = oracle_payload.get(SUPPORT_KEY)
            propagation_status = normalize_relation_status(
                propagation_entries.get(relation, {}).get(STATUS_KEY)
            )

            if oracle_status is None:
                none_after_oracle_total += 1
                if propagation_status == STATUS_INFERRED:
                    none_to_inferred_count += 1

            if oracle_status == STATUS_OBSERVED:
                observed_after_oracle_total += 1
                if propagation_status == STATUS_OBSERVED_AND_INFERRED:
                    observed_to_observed_and_inferred_count += 1

            if isinstance(oracle_support, (int, float)) and oracle_support > 0:
                oracle_has_positive_support = True

        for propagation_payload in propagation_entries.values():
            propagation_support = propagation_payload.get(SUPPORT_KEY)
            if isinstance(propagation_support, (int, float)) and propagation_support > 0:
                propagation_has_positive_support = True
                break

        if not oracle_has_positive_support:
            domains_without_positive_support_after_oracle += 1
            if propagation_has_positive_support:
                domains_without_positive_support_to_positive_after_propagation += 1

    return {
        "none_after_oracle_total": none_after_oracle_total,
        "none_to_inferred_count": none_to_inferred_count,
        "observed_after_oracle_total": observed_after_oracle_total,
        "observed_to_observed_and_inferred_count": observed_to_observed_and_inferred_count,
        "initial_total_relations": count_total_relations(initial_qcn),
        "after_propagation_total_relations": count_total_relations(after_propagation_qcn),
        "domains_without_positive_support_after_oracle": domains_without_positive_support_after_oracle,
        "domains_without_positive_support_to_positive_after_propagation": domains_without_positive_support_to_positive_after_propagation,
    }


def processReductionRates(initialQCN, QCNAfterOracle, QCNAfterPC):
    """
    Calcule le nombre cumulé de relations d'Allen dans trois QCN pour analyser les taux de réduction.

    :param initialQCN: QCN initial (domaines complets)
    :param QCNAfterOracle: QCN après interrogation de l'oracle
    :param QCNAfterPC: QCN après path consistency
    :return: dict contenant les statistiques de réduction
    """

    # Calculer les statistiques
    initial_count = count_total_relations(initialQCN)
    after_oracle_count = count_total_relations(QCNAfterOracle)
    after_pc_count = count_total_relations(QCNAfterPC)

    # Calculer les taux de réduction
    oracle_reduction = (
        ((initial_count - after_oracle_count) / initial_count * 100)
        if initial_count > 0
        else 0
    )
    pc_reduction = (
        ((after_oracle_count - after_pc_count) / after_oracle_count * 100)
        if after_oracle_count > 0
        else 0
    )
    total_reduction = (
        ((initial_count - after_pc_count) / initial_count * 100)
        if initial_count > 0
        else 0
    )

    reductionRates = {
        "initial_total_relations": initial_count,
        "after_oracle_total_relations": after_oracle_count,
        "after_pc_total_relations": after_pc_count,
        "oracle_reduction_percent": oracle_reduction,
        "pc_reduction_percent": pc_reduction,
        "total_reduction_percent": total_reduction,
        "num_constraints": len(initialQCN),
    }

    return reductionRates


def save_qcn_to_file(learning_result: dict, kg_type: str):
    """
    Sauvegarde les résultats de l'apprentissage dans 2 fichiers :
    1. Un fichier JSON avec les QCNs (initial, après oracle, après propagation)
    2. Un fichier texte avec les statistiques de réduction

    :param learning_result: dict contenant 'initial', 'after_oracle', 'after_propagation', 'stats'
    :param kg_type: type du knowledge graph
    """
    # Créer le dossier Results/Q.../ s'il n'existe pas
    results_dir = results_root / kg_type
    os.makedirs(results_dir, exist_ok=True)

    # Noms des fichiers
    output_stem = build_output_stem(kg_type)
    qcn_filename = f"{output_stem}.json"
    stats_filename = f"{output_stem}_stats.txt"
    qcn_path = results_dir / qcn_filename
    stats_path = results_dir / stats_filename

    # 1. Sauvegarder les QCNs en JSON
    def convert_qcn_for_json(qcn):
        """Convertit le QCN en format JSON-compatible"""
        return {f"{k[0]}___{k[1]}": v for k, v in qcn.items()}

    qcn_data = {
        "initial": convert_qcn_for_json(learning_result["initial"]),
        "after_oracle": convert_qcn_for_json(learning_result["after_oracle"]),
    }
    if "after_propagation" in learning_result:
        qcn_data["after_propagation"] = convert_qcn_for_json(
            learning_result["after_propagation"]
        )

    with qcn_path.open("w", encoding="UTF-8") as f:
        json.dump(qcn_data, f, indent=2, ensure_ascii=False)

    print(f"✅ QCNs saved to: {qcn_path}")

    # 2. Sauvegarder les statistiques en format texte lisible
    stats = learning_result.get("stats")
    oracle_stats = learning_result.get("oracle_stats", {})
    property_filter_stats = learning_result.get("property_filter_stats", {})
    qcn_stats = learning_result.get("qcn_stats", {})
    after_oracle_report_stats = learning_result.get("after_oracle_report_stats", {})
    after_propagation_qcn_stats = learning_result.get("after_propagation_qcn_stats", {})
    oracle_to_propagation_transition_stats = learning_result.get(
        "oracle_to_propagation_transition_stats", {}
    )
    timing = learning_result.get("timing", {})

    def pct(part: int, total: int) -> float:
        return (part / total * 100.0) if total > 0 else 0.0

    with stats_path.open("w", encoding="UTF-8") as f:
        f.write(f"# Reduction statistics - Knowledge Graph: {kg_type}\n")
        f.write("# Mode: Learner2 (threshold-free)\n")
        f.write("#" + "=" * 70 + "\n\n")

        if stats is not None:
            f.write("## TOTAL NUMBER OF ALLEN RELATIONS\n")
            f.write(
                f"Initial (before learning)        : {stats['initial_total_relations']:>8}\n"
            )
            f.write(
                f"After Oracle                     : {stats['after_oracle_total_relations']:>8}\n"
            )
            f.write(
                f"After Path Consistency           : {stats['after_pc_total_relations']:>8}\n"
            )
            f.write("\n")

            f.write("## REDUCTION RATES\n")
            f.write(
                f"Reduction by Oracle              : {stats['oracle_reduction_percent']:>7.2f}%\n"
            )
            f.write(
                f"Reduction by Path Consistency    : {stats['pc_reduction_percent']:>7.2f}%\n"
            )
            f.write(
                f"Total Reduction                  : {stats['total_reduction_percent']:>7.2f}%\n"
            )
            f.write("\n")

            f.write("## ADDITIONAL INFORMATION\n")
            f.write(f"Number of constraints (pairs)    : {stats['num_constraints']:>8}\n")

            # Relations moyennes par contrainte
            avg_initial = (
                stats["initial_total_relations"] / stats["num_constraints"]
                if stats["num_constraints"] > 0
                else 0
            )
            avg_oracle = (
                stats["after_oracle_total_relations"] / stats["num_constraints"]
                if stats["num_constraints"] > 0
                else 0
            )
            avg_pc = (
                stats["after_pc_total_relations"] / stats["num_constraints"]
                if stats["num_constraints"] > 0
                else 0
            )
            f.write("Average relations per constraint\n")
            f.write(f"  - Initial                      : {avg_initial:>7.2f}\n")
            f.write(f"  - After Oracle                 : {avg_oracle:>7.2f}\n")
            f.write(f"  - After PC                     : {avg_pc:>7.2f}\n")
            if "cpu_time_seconds" in stats:
                f.write("\n")
                f.write("## PERFORMANCE\n")
                f.write(
                    f"CPU time (learner2)              : {stats['cpu_time_seconds']:>8.4f} s\n"
                )
        else:
            f.write("## AVAILABLE STATISTICS (ORACLE PHASE)\n")
            if qcn_stats:
                f.write(f"Domains (pairs)                 : {qcn_stats.get('num_domains', 0):>8}\n")
                f.write(f"Total relations                 : {qcn_stats.get('total_relations', 0):>8}\n")
                f.write(
                    "Observed rel. & support > 0    : "
                    f"{(qcn_stats.get('count_observed', 0) + qcn_stats.get('count_observed_and_inferred', 0)):>8}\n"
                )
                f.write(
                    "  -> proportion                : "
                    f"{(qcn_stats.get('count_observed', 0) + qcn_stats.get('count_observed_and_inferred', 0))} / "
                    f"{qcn_stats.get('total_relations', 0)} "
                    f"({qcn_stats.get('percent_observed_sup_pos', 0):.2f}%)\n"
                )
                f.write(
                    "Rel. status=None & sup=None    : "
                    f"{qcn_stats.get('count_none_none', 0):>8}\n"
                )
                f.write(
                    "  -> proportion                : "
                    f"{qcn_stats.get('count_none_none', 0)} / "
                    f"{qcn_stats.get('total_relations', 0)} "
                    f"({qcn_stats.get('percent_none_none', 0):.2f}%)\n"
                )
                f.write(
                    "Rel. status=None & sup == 0    : "
                    f"{qcn_stats.get('count_none_zero', 0):>8}\n"
                )
                f.write(
                    "  -> proportion                : "
                    f"{qcn_stats.get('count_none_zero', 0)} / "
                    f"{qcn_stats.get('total_relations', 0)} "
                    f"({qcn_stats.get('percent_none_zero', 0):.2f}%)\n"
                )
                f.write(
                    "Inferred relations             : "
                    f"{qcn_stats.get('count_inferred', 0):>8}\n"
                )
                f.write(
                    "Observed and inferred relations: "
                    f"{qcn_stats.get('count_observed_and_inferred', 0):>8}\n"
                )
                f.write(
                    "% domains without observed relation: "
                    f"{qcn_stats.get('count_domains_without_observed', 0)} / "
                    f"{qcn_stats.get('num_domains', 0)} "
                    f"({qcn_stats.get('percent_domains_without_observed', 0):.2f}%)\n"
                )
                f.write(
                    "% domains with >=1 observed rel. & sup>0: "
                    f"{qcn_stats.get('count_domains_with_observed_sup_pos', 0)} / "
                    f"{qcn_stats.get('num_domains', 0)} "
                    f"({qcn_stats.get('percent_domains_with_observed_sup_pos', 0):.2f}%)\n"
                )
                f.write("Support>0 per domain (avg/min/max/std) : ")
                f.write(
                    f"{qcn_stats.get('avg_support_per_domain', 0):.2f}/"
                    f"{qcn_stats.get('min_support_per_domain', 0)}/"
                    f"{qcn_stats.get('max_support_per_domain', 0)}/"
                    f"{qcn_stats.get('std_support_per_domain', 0):.2f}\n"
                )
            if oracle_stats:
                f.write(f"Oracle calls                    : {oracle_stats.get('total_calls', 0):>8}\n")
                f.write(
                    f"Total oracle phase cost (s)     : {oracle_stats.get('cost_of_the_oracle_phase', 0):>8.4f}\n"
                )
                f.write(
                    f"Oracle time min/max/avg (s)     : {oracle_stats.get('min_cpu_time', 0):.6f} / "
                    f"{oracle_stats.get('max_cpu_time', 0):.6f} / "
                    f"{oracle_stats.get('avg_cpu_time', 0):.6f}\n"
                )

        if oracle_stats:
            f.write("\n## TEMPS CPU - PHASE ORACLE\n")
            f.write(
                f"Appels oracle              : {int(oracle_stats.get('total_calls', 0))}\n"
            )
            f.write(
                f"Coût total phase oracle    : {float(oracle_stats.get('cost_of_the_oracle_phase', 0.0)):.4f} s\n"
            )
            f.write(
                "Temps oracle / appel - moy : "
                f"{float(oracle_stats.get('avg_cpu_time', 0.0)):.6f} s  "
                f"min={float(oracle_stats.get('min_cpu_time', 0.0)):.6f} s  "
                f"max={float(oracle_stats.get('max_cpu_time', 0.0)):.6f} s\n"
            )

        if property_filter_stats:
            f.write("\n## FILTRAGE PROPRIETES (TOP-K)\n")
            f.write(
                "Fraction top-k (proprietes)   : "
                f"{float(property_filter_stats.get('top_k_fraction', 0.0)):.2f}\n"
            )
            f.write(
                f"Proprietes gardees            : {int(property_filter_stats.get('kept_properties_count', 0))}\n"
            )
            f.write(
                f"Faits gardes                  : {int(property_filter_stats.get('kept_facts_count', 0))}\n"
            )

        if after_oracle_report_stats:
            oracle_total_rel = int(after_oracle_report_stats.get("total_relations", 0))
            oracle_total_dom = int(after_oracle_report_stats.get("total_domains", 0))
            observed_support_pos = int(
                after_oracle_report_stats.get("observed_support_pos_relations", 0)
            )
            zero_support = int(after_oracle_report_stats.get("zero_support_relations", 0))
            dom_with_observed = int(
                after_oracle_report_stats.get("domains_with_observed_status", 0)
            )
            dom_with_none = int(
                after_oracle_report_stats.get("domains_with_none_support", 0)
            )

            f.write("\n## AFTER ORACLE\n")
            f.write(
                "Relations support > 0 (Observed) : "
                f"{observed_support_pos} / {oracle_total_rel} "
                f"({pct(observed_support_pos, oracle_total_rel):.2f}%)\n"
            )
            f.write(
                "Relations support == 0           : "
                f"{zero_support} / {oracle_total_rel} "
                f"({pct(zero_support, oracle_total_rel):.2f}%)\n"
            )
            f.write(
                "Domains with >=1 Observed status : "
                f"{dom_with_observed} / {oracle_total_dom} "
                f"({pct(dom_with_observed, oracle_total_dom):.2f}%)\n"
            )
            f.write(
                "Domains with None support        : "
                f"{dom_with_none} / {oracle_total_dom} "
                f"({pct(dom_with_none, oracle_total_dom):.2f}%)\n"
            )

        if after_propagation_qcn_stats:
            prop_total_rel = int(after_propagation_qcn_stats.get("total_relations", 0))
            prop_observed = int(after_propagation_qcn_stats.get("count_observed", 0))
            prop_observed_inferred = int(
                after_propagation_qcn_stats.get("count_observed_and_inferred", 0)
            )
            prop_inferred = int(after_propagation_qcn_stats.get("count_inferred", 0))

            f.write("\n## AFTER PROPAGATION\n")
            f.write(
                "Relations status Observed        : "
                f"{prop_observed} / {prop_total_rel} "
                f"({pct(prop_observed, prop_total_rel):.2f}%)\n"
            )
            f.write(
                "Relations status Obs.+Inferred   : "
                f"{prop_observed_inferred} / {prop_total_rel} "
                f"({pct(prop_observed_inferred, prop_total_rel):.2f}%)\n"
            )
            f.write(
                "Relations status Inferred        : "
                f"{prop_inferred} / {prop_total_rel} "
                f"({pct(prop_inferred, prop_total_rel):.2f}%)\n"
            )

        if oracle_to_propagation_transition_stats:
            none_after_oracle_total = int(
                oracle_to_propagation_transition_stats.get(
                    "none_after_oracle_total", 0
                )
            )
            none_to_inferred_count = int(
                oracle_to_propagation_transition_stats.get(
                    "none_to_inferred_count", 0
                )
            )
            observed_after_oracle_total = int(
                oracle_to_propagation_transition_stats.get(
                    "observed_after_oracle_total", 0
                )
            )
            observed_to_observed_and_inferred_count = int(
                oracle_to_propagation_transition_stats.get(
                    "observed_to_observed_and_inferred_count", 0
                )
            )
            initial_total_relations = int(
                oracle_to_propagation_transition_stats.get(
                    "initial_total_relations", 0
                )
            )
            after_propagation_total_relations = int(
                oracle_to_propagation_transition_stats.get(
                    "after_propagation_total_relations", 0
                )
            )
            domains_without_positive_support_after_oracle = int(
                oracle_to_propagation_transition_stats.get(
                    "domains_without_positive_support_after_oracle", 0
                )
            )
            domains_without_positive_support_to_positive_after_propagation = int(
                oracle_to_propagation_transition_stats.get(
                    "domains_without_positive_support_to_positive_after_propagation",
                    0,
                )
            )

            f.write("\n## TRANSITIONS ORACLE -> PROPAGATION\n")
            f.write(
                "Relations None -> Inferred       : "
                f"{none_to_inferred_count} / {none_after_oracle_total} "
                f"({pct(none_to_inferred_count, none_after_oracle_total):.2f}%)\n"
            )
            f.write(
                "Relations Observed -> Obs.+Inf. : "
                f"{observed_to_observed_and_inferred_count} / {observed_after_oracle_total} "
                f"({pct(observed_to_observed_and_inferred_count, observed_after_oracle_total):.2f}%)\n"
            )
            f.write(
                "Total relations in initial QCN  : "
                f"{initial_total_relations}\n"
            )
            f.write(
                "Total relations after propagation: "
                f"{after_propagation_total_relations}\n"
            )
            f.write(
                "Domains without sup>0 -> positive: "
                f"{domains_without_positive_support_to_positive_after_propagation} / "
                f"{domains_without_positive_support_after_oracle} "
                f"({pct(domains_without_positive_support_to_positive_after_propagation, domains_without_positive_support_after_oracle):.2f}%)\n"
            )

        if timing:
            f.write("\n## CPU TIME PER STAGE\n")
            f.write("Initial stage (reference)       :   0.0000 s\n")
            f.write(
                f"Stage after_oracle (s)          : {timing.get('oracle_phase_cpu_time', 0):>8.4f}\n"
            )
            f.write(
                f"Stage after_propagation (s)     : {timing.get('propagate_and_filter_cpu_time', 0):>8.4f}\n"
            )
            f.write("\n## CUMULATIVE CPU TIME PER STAGE\n")
            f.write("Cumulative after initial (s)    :   0.0000 s\n")
            f.write(
                f"Cumulative after after_oracle(s): {timing.get('cumulative_after_oracle_cpu_time', 0):>8.4f}\n"
            )
            f.write(
                f"Cumulative after after_propagation(s): {timing.get('cumulative_after_propagation_cpu_time', 0):>8.4f}\n"
            )
            f.write("\n## GLOBAL CPU TIME\n")
            f.write(
                f"Total learner2 (s)              : {timing.get('learner_total_cpu_time', 0):>8.4f}\n"
            )

    print(f"✅ Statistics saved to: {stats_path}")


def learner_with_timeout(
    entity_network,
    entities,
    properties,
    kgName,
    timeout=600,
):
    """
    Lance l'apprentissage avec timeout et sauvegarde les résultats.

    :param entity_network: réseau de contraintes initial (QCN)
    :param entities: dictionnaire des entités
    :param properties: ensemble/liste des propriétés
    :param kgName: type du knowledge graph
    :param timeout: timeout en secondes
    """
    with mp.Pool(processes=1) as pool:
        result = pool.apply_async(
            learner2,
            (entity_network, entities, properties),
        )
        try:
            learning_result = result.get(timeout=timeout)
            if learning_result is not None:
                # Sauvegarder les QCNs et les statistiques
                save_qcn_to_file(learning_result, kgName)
            return learning_result
        except mp.TimeoutError:
            print(f"⏱️ Timeout : calcul interrompu après {timeout / 60} minutes")
            pool.terminate()
            pool.join()
            print("⚠️ Apprentissage interrompu - aucun résultat sauvegardé")
            return None


def Query(Vocabulary, entities):
    """
    Vérifié
    Docstring for Query

    :param Vocabulary: Description
    :param entities: Description
    :param threshold: Description
    :return: Description
    :rtype: bool
    """

    p1, p2 = Vocabulary

    allenrelMatching: dict[str, float] = {}

    for r in (
        ALLEN_RELATIONS
    ):  # initialisation du compteur de correspondances pour chaque relation d'Allen
        allenrelMatching[r] = 0
    nbr_total = 0

    for head in entities:
        properties_of_head = entities[head].triples_per_p.keys()
        if p1 not in properties_of_head or p2 not in properties_of_head:
            continue  # Skip this entity if it doesn't have both properties
        else:  # Si l'entité est décrite par les deux propriétés
            quadruplet_p1 = entities[head].triples_per_p.get(p1)
            quadruplet_p2 = entities[head].triples_per_p.get(p2)
            for qp1 in quadruplet_p1:
                for qp2 in quadruplet_p2:
                    nbr_total += 1
                    answers = AllenR.AllenRelation(
                        qp1, qp2
                    ).check_all_axioms()  # vérification de toutes les relations d'Allen entre les deux quadruplets
                    for (
                        r,
                        holds,
                    ) in answers.items():  # mise à jour du compteur de correspondances pour chaque relation d'Allen
                        if holds:
                            allenrelMatching[r] += 1

    if nbr_total == 0.0:
        # Aucun couple de faits (p1, p2) observé dans le graphe: support/status restent à None.
        no_evidence_supports = {relation: None for relation in ALLEN_RELATIONS}
        return with_relation_data(
            no_evidence_supports,
            {relation: None for relation in ALLEN_RELATIONS},
        )
    else:
        # Retourner les scores (proportion d'entités vérifiant chaque relation)
        for r in ALLEN_RELATIONS:
            allenrelMatching[r] = allenrelMatching[r] / nbr_total
        return with_relation_data(
            allenrelMatching,
            {
                relation: relation_status(
                    observed=allenrelMatching[relation] > 0,
                    inferred=False,
                )
                for relation in ALLEN_RELATIONS
            },
        )


def analyze_qcn_scores(kg_type: str) -> dict | None:
    """
    Analyse un fichier JSON QCN pour compter les pourcentages de relations par catégorie de score.
    
    :param kg_type: type du knowledge graph (Q6256, Q215380, Q82955)
    :return: dict contenant les statistiques par QCN (after_oracle, after_propagation)
    """
    qcn_path = results_root / kg_type / f"{build_output_stem(kg_type)}.json"
    
    if not qcn_path.exists():
        print(f"Fichier non trouvé: {qcn_path}")
        return None
    
    with qcn_path.open("r", encoding="UTF-8") as f:
        qcn_data = json.load(f)
    
    results = {}
    
    for qcn_name in ["after_oracle", "after_propagation"]:
        if qcn_name not in qcn_data:
            continue
            
        qcn = qcn_data[qcn_name]
        
        total_relations = 0
        count_repaired = 0  # score == -1
        count_no_entities = 0  # score == -2
        count_observed = 0  # score > 0
        count_zero = 0  # score == 0
        
        # Parcourir toutes les paires dans le QCN
        for pair_key, domain in qcn.items():
            for relation, score in relation_scores(domain).items():
                total_relations += 1
                
                if score == -1.0:
                    count_repaired += 1
                elif score == -2.0:
                    count_no_entities += 1
                elif score > 0:
                    count_observed += 1
                elif score == 0:
                    count_zero += 1
        
        # Calculer les pourcentages
        if total_relations > 0:
            results[qcn_name] = {
                "total_relations": total_relations,
                "repaired_count": count_repaired,
                "repaired_percent": (count_repaired / total_relations) * 100,
                "no_entities_count": count_no_entities,
                "no_entities_percent": (count_no_entities / total_relations) * 100,
                "observed_count": count_observed,
                "observed_percent": (count_observed / total_relations) * 100,
                "zero_count": count_zero,
                "zero_percent": (count_zero / total_relations) * 100,
            }
    
    return results


def print_score_analysis(kg_type: str):
    """
    Affiche l'analyse des scores pour un KG donné.
    
    :param kg_type: type du knowledge graph
    """
    results = analyze_qcn_scores(kg_type)
    
    if not results:
        return
    
    print(f"\n{'=' * 70}")
    print(f"ANALYSE DES SCORES - Knowledge Graph: {kg_type}")
    print(f"{'=' * 70}\n")
    
    for qcn_name, stats in results.items():
        print(f"📊 {qcn_name.upper().replace('_', ' ')}")
        print(f"   Total de relations: {stats['total_relations']}")
        print(f"   Relations observées (score > 0):  {stats['observed_count']:>6} ({stats['observed_percent']:>6.2f}%)")
        print(f"   Relations réparées (score = -1): {stats['repaired_count']:>6} ({stats['repaired_percent']:>6.2f}%)")
        print(f"   Aucune entité (score = -2):      {stats['no_entities_count']:>6} ({stats['no_entities_percent']:>6.2f}%)")
        if stats['zero_count'] > 0:
            print(f"   Score zéro (score = 0):           {stats['zero_count']:>6} ({stats['zero_percent']:>6.2f}%)")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QCN Generator 2 - Apprentissage de contraintes temporelles (threshold-free)"
    )
    parser.add_argument("kg", type=str, help="Knowledge Graph (Q6256, Q215380, Q82955)")
    parser.add_argument(
        "timeout", type=int, help="Timeout en secondes (ex: 600 pour 10 min)"
    )

    args = parser.parse_args()
    kg_name = args.kg
    timeout_seconds = args.timeout

    print(
        "-----------------------QCNGenerator2.py (threshold-free)--------------------------"
    )
    print(f"Knowledge Graph: {kg_name}")
    print(f"Timeout: {timeout_seconds}s ({timeout_seconds / 60:.1f} min)")
    print(f"{'=' * 70}")

    initial_entities, initial_properties = read_KG(kg_name)
    entities = copy.deepcopy(initial_entities)
    properties = copy.deepcopy(initial_properties)

    entity_network = build_entity_network(
        properties=properties,
        default_constraints=ALLEN_RELATIONS,
        top_k_fraction=PROPERTY_SUPPORT_THRESHOLD,
        entities=entities,
    )

    # Keep only properties actually retained by support filtering.
    properties = sorted({p for pair in entity_network.keys() for p in pair})
    kept_properties = set(properties)
    kept_facts_count = 0
    for entity in entities.values():
        for prop, triples in entity.triples_per_p.items():
            if prop in kept_properties:
                kept_facts_count += len(triples)

    print(
        f"Top {PROPERTY_SUPPORT_THRESHOLD:.0%} properties by quadruplet count: "
        f"{len(initial_properties)} -> {len(properties)} properties kept"
    )
    print(f"Facts kept after top-{PROPERTY_SUPPORT_THRESHOLD:.0%} filter: {kept_facts_count}")

    print("\n🚀 Début de l'apprentissage des contraintes temporelles...")
    learned_entity_network = learner_with_timeout(
        entity_network,
        entities,
        properties,
        kg_name,
        timeout=timeout_seconds,
    )

    if learned_entity_network is not None:
        print(f"✅ Apprentissage terminé avec succès pour {kg_name}")
        # Afficher l'analyse des scores
        print_score_analysis(kg_name)
    else:
        print(f"⚠️ Apprentissage interrompu par timeout pour {kg_name}")

    print(f"{'=' * 70}")
    print("🏁 EXPÉRIENCE TERMINÉE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()