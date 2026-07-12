#!/usr/bin/env python3
"""Export English UMLS concepts and selected relations as Neo4j CSV files.

The input files are UMLS RRF exports.  Paths are deliberately supplied at
runtime so that local data locations and licensed-resource details are not
embedded in the repository.
"""

import argparse
import csv
from pathlib import Path


EXCLUDED_RELATIONS = {
    "translation_of",
    "inverse_isa",
    "isa",
    "ro",
    "rb",
    "rn",
    "rq",
    "par",
    "chd",
    "member_of",
    "concept_in_subset",
    "mapped_to",
    "mapped_from",
    "primary_mapped_to",
    "primary_mapped_from",
    "same_as",
    "classified_as",
    "sy",
    "use",
    "qb",
    "spectinomycin titr sbt",
    "has_time_aspect",
    "time_aspect_of",
    "has_scale",
    "scale_of",
    "has_temporal_context",
    "permuted_term_of",
    "has_permuted_term",
    "has_translation",
    "entry_version_of",
}


def read_english_entities(path: Path) -> dict[str, str]:
    """Return the first English preferred string found for each CUI."""
    entities: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            fields = line.rstrip("\n").split("|")
            if len(fields) <= 14:
                continue
            cui, language, name = fields[0].strip(), fields[1].strip(), fields[14].strip()
            if language == "ENG" and cui and cui not in entities:
                entities[cui] = name
    return entities


def read_relations(path: Path, entities: set[str]) -> list[tuple[str, str, str]]:
    """Return non-excluded relations whose endpoints are English CUIs."""
    edges: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            fields = line.rstrip("\n").split("|")
            if len(fields) <= 7:
                continue
            start_cui = fields[0].strip()
            end_cui = fields[4].strip()
            relation = fields[7].strip()
            if (
                relation
                and relation.casefold() not in EXCLUDED_RELATIONS
                and start_cui in entities
                and end_cui in entities
            ):
                edges.append((start_cui, relation, end_cui))
    return edges


def write_csv_files(output_dir: Path, entities: dict[str, str], edges: list[tuple[str, str, str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "nodes.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["CUI", "name"])
        writer.writerows(entities.items())

    with (output_dir / "relations.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["start_CUI", "rel", "end_CUI"])
        writer.writerows(edges)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mrconso", type=Path, required=True, help="Path to MRCONSO.RRF")
    parser.add_argument("--mrrel", type=Path, required=True, help="Path to MRREL.RRF")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for nodes.csv and relations.csv")
    args = parser.parse_args()

    entities = read_english_entities(args.mrconso)
    edges = read_relations(args.mrrel, set(entities))
    write_csv_files(args.output_dir, entities, edges)
    print(f"Exported {len(entities)} nodes and {len(edges)} relations to {args.output_dir}")


if __name__ == "__main__":
    main()
