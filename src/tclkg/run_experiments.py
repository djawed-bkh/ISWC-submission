#!/usr/bin/env python3
"""
Script pour lancer plusieurs experiences Learner2 en parallele.
"""

import subprocess
import sys
import os
import argparse
from datetime import datetime
import multiprocessing as mp
from pathlib import Path
from typing import Tuple

# ============ CONFIGURATION DES EXPÉRIENCES ============

timeout = 86400  # 24 heures de timeout pour chaque expérience

# Configurations pour QCNGenerator2.py (learner2 threshold-free)
# Format principal: (kg, timeout, repair_inconsistency, discover)
# Les formats legacy restent supportes: (kg, timeout) et (kg, timeout, repair)
EXPERIMENTS_GEN2 = [
    # KG Q6256
    # ("Q6256", timeout, False, False),  # Sans réparation, sans découverte
    ("Q6256", timeout, True, False),  # Avec réparation
    # KG Q215380
    # ("Q215380", timeout, False, False),
    ("Q215380", timeout, True, False),
    # KG Q82955
    # ("Q82955", timeout, False, False),
    # ("Q82955", timeout, True, False), # cette expe doit etre lancée toute seule car prend beaucoup de mémoire
]

# Nombre d'expe à lancer simultanément
MAX_PARALLEL = 6

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Dossier pour les logs
LOG_DIR = PROJECT_ROOT / "logs_experiments"

# ============================================================


def run_single_experiment(config: Tuple) -> Tuple:
    """
    Lance une seule experience Learner2 avec les parametres donnes.
    Retourne (kg, timeout, return_code, repair, discover)
    """
    generator_version = 2
    repair = False
    discover = False
    if len(config) == 4:
        kg, timeout, repair, discover = config
    elif len(config) == 3:
        kg, timeout, repair = config
    elif len(config) == 2:
        kg, timeout = config
    else:
        raise ValueError(f"Unsupported learner2 config format: {config}")

    # Créer le dossier de logs s'il n'existe pas
    os.makedirs(LOG_DIR, exist_ok=True)

    # Nom du fichier log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    repair_str = "repair" if repair else "norepair"
    discover_str = "discover" if discover else "nodiscover"
    log_file = (
        LOG_DIR
        / f"exp_{kg}_gen{generator_version}_{repair_str}_{discover_str}_{timestamp}.log"
    )
    print(
        f"Launch: KG={kg}, timeout={timeout}s, repair={repair}, discover={discover}"
    )
    print(f"   Log: {log_file}")

    try:
        with log_file.open("w", encoding="utf-8") as log:
            log.write(f"Experience: KG={kg}, Timeout={timeout}s\n")
            log.write(f"Démarrage: {datetime.now()}\n")
            log.write("=" * 70 + "\n\n")
            log.flush()

            python_executable = sys.executable

            cmd_args = [
                python_executable,
                "-u",
                "-m",
                "tclkg.qcn_generator2",
                kg,
                str(timeout),
            ]
            if repair:
                cmd_args.append("--repair")
            if discover:
                cmd_args.append("--discover")

            process = subprocess.Popen(
                cmd_args,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=PROJECT_ROOT,
                bufsize=1,
                universal_newlines=True,
            )

            return_code = process.wait()

            log.write(f"\n\n{'=' * 70}\n")
            log.write(f"Fin: {datetime.now()}\n")
            log.write(f"Code de retour: {return_code}\n")

        if return_code == 0:
            print(f"OK: KG={kg}, repair={repair_str}, discover={discover_str}")
        else:
            print(
                f"ERROR: KG={kg}, repair={repair_str}, discover={discover_str} (code={return_code})"
            )

        return (kg, timeout, return_code, repair, discover)

    except Exception as e:
        print(f"Exception for KG={kg}: {e}")
        import traceback

        traceback.print_exc()
        return (kg, timeout, -1, repair, discover)


def main():
    parser = argparse.ArgumentParser(
        description="Lancer plusieurs experiences Learner2 en parallele"
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=MAX_PARALLEL,
        help="Nombre maximum d'experiences executees en parallele.",
    )
    parser.add_argument(
        "--kg",
        type=str,
        default=None,
        help="Filtre optionnel pour executer un seul KG (ex: Q6256).",
    )
    args = parser.parse_args()

    experiments = EXPERIMENTS_GEN2
    if args.kg:
        experiments = [cfg for cfg in experiments if cfg[0] == args.kg]
        if not experiments:
            raise ValueError(f"No learner2 experiment configured for KG={args.kg}")

    print("=" * 70)
    print("Launching learner2 experiments")
    print("=" * 70)
    print("Generator: tclkg.qcn_generator2")
    print(f"Nombre d'expériences: {len(experiments)}")
    print(f"Executions simultanees: {args.max_parallel}")
    print(f"Logs dans: {LOG_DIR}/")
    print("=" * 70)

    start_time = datetime.now()

    with mp.Pool(processes=args.max_parallel) as pool:
        results = pool.map(run_single_experiment, experiments)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Résumé
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    success = 0
    failed = 0

    for result in results:
        kg, timeout, code, repair, discover = result

        status = "OK" if code == 0 else f"ERROR (code={code})"
        repair_info = "repair" if repair else "norepair"
        discover_info = "discover" if discover else "nodiscover"
        print(
            f"  KG={kg:<10} {repair_info:<10} {discover_info:<10} timeout={timeout:>6}s  {status}"
        )
        if code == 0:
            success += 1
        else:
            failed += 1

    print("=" * 70)
    print(f"Success: {success}/{len(experiments)}")
    print(f"Failed: {failed}/{len(experiments)}")
    print(f"Total duration: {duration / 60:.1f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()
