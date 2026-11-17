import csv
import json
import hashlib
import shutil
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "CICIoT2023_Selected"
RUN_DIR = PROJECT_ROOT / "runs" / "experiment_001_dataset_char"
CLEANED_DIR = RUN_DIR / "cleaned_data"

CHUNK_SIZE = 100_000
RANDOM_SEED = 20261002

EXPECTED_TOTAL = 10_713_192

EXPECTED_ROWS = {
    "train": 7_702_608,
    "val": 1_236_682,
    "test": 1_773_902,
}

EXPECTED_FILES = 46

EXPECTED_SPLITS = (
    "train",
    "val",
    "test",
)

EXPECTED_COLUMNS = [
    "Header_Length",
    "Protocol Type",
    "Time_To_Live",
    "Rate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IGMP",
    "IPv",
    "LLC",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT",
    "Number",
    "Variance",
]

CATEGORY_MAP = {
    "Benign_Final": "Benign",
    "DDoS-HTTP_Flood": "DDoS",
    "DDoS-SlowLoris": "DDoS",
    "DDoS-SYN_Flood": "DDoS",
    "DDoS-UDP_Flood": "DDoS",
    "Recon-PortScan": "Recon",
    "SqlInjection": "Web",
    "XSS": "Web",
}


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def count_csv_rows(path):
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)

        return sum(1 for _ in reader)


def read_header(path):
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)

        return next(reader, None)


def get_category(source_folder):
    if source_folder not in CATEGORY_MAP:
        raise RuntimeError(
            f"Unknown source folder '{source_folder}'. "
            f"Known folders: {sorted(CATEGORY_MAP)}"
        )

    return CATEGORY_MAP[source_folder]


def deterministic_key(relative_path):
    value = f"{RANDOM_SEED}:{relative_path}"

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def discover_files():
    files = sorted(RAW_DIR.rglob("*.csv"))

    if len(files) != EXPECTED_FILES:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_FILES} CSV files, "
            f"found {len(files)}."
        )

    return files


def validate_schemas(files):
    print("[2/9] Validating schemas...")

    reference = None

    for path in files:
        header = read_header(path)

        if header is None:
            raise RuntimeError(f"Empty CSV file: {path}")

        if len(header) != 39:
            raise RuntimeError(
                f"{path} has {len(header)} columns instead of 39."
            )

        if header != EXPECTED_COLUMNS:
            raise RuntimeError(f"Schema mismatch in {path}.")

        if reference is None:
            reference = header
        elif header != reference:
            raise RuntimeError(
                f"Schema differs from reference in {path}."
            )

    print("    Columns: 39")
    print("    Features: 39")
    print("    Schema identical across all files: YES")


def build_manifest(files):
    print("[3/9] Building source file manifest...")

    manifest_files = []

    for path in files:
        relative_path = str(path.relative_to(RAW_DIR))

        source_folder = path.parent.name
        category = get_category(source_folder)
        row_count = count_csv_rows(path)

        if row_count <= 0:
            raise RuntimeError(
                f"CSV contains no data rows: {relative_path}"
            )

        manifest_files.append(
            {
                "file": relative_path,
                "source_folder": source_folder,
                "original_attack": source_folder,
                "category": category,
                "row_count": row_count,
                "column_count": 39,
                "sha256": sha256_file(path),
                "synthetic_data": False,
                "mock_data": False,
                "sampling": False,
                "label_source": "parent_directory",
            }
        )

    manifest_files.sort(key=lambda item: item["file"])

    return manifest_files


def validate_source_dataset(manifest_files):
    print("[4/9] Validating source dataset...")

    total_rows = sum(item["row_count"] for item in manifest_files)

    if total_rows != EXPECTED_TOTAL:
        raise RuntimeError(
            "Raw dataset total mismatch.\n"
            f"Expected: {EXPECTED_TOTAL:,}\n"
            f"Found:    {total_rows:,}"
        )

    categories = Counter(item["category"] for item in manifest_files)

    print(f"    Source files: {len(manifest_files)}")
    print(f"    Source rows:  {total_rows:,}")
    print("    Categories:")

    for category in sorted(categories):
        rows = sum(
            item["row_count"]
            for item in manifest_files
            if item["category"] == category
        )

        print(
            f"        {category:10} "
            f"{categories[category]:2} files "
            f"{rows:,} rows"
        )

    required_categories = {"Benign", "DDoS", "Recon", "Web"}

    missing = required_categories - set(categories)

    if missing:
        raise RuntimeError(
            "Required actual categories are missing: "
            f"{sorted(missing)}"
        )

    print("    All four actual categories present: YES")

    return categories


def calculate_category_targets(manifest_files):
    category_totals = defaultdict(int)

    for item in manifest_files:
        category_totals[item["category"]] += item["row_count"]

    targets = {}

    for category in sorted(category_totals):
        total = category_totals[category]

        raw = {
            split: total * EXPECTED_ROWS[split] / EXPECTED_TOTAL
            for split in EXPECTED_SPLITS
        }

        values = {split: int(raw[split]) for split in EXPECTED_SPLITS}

        remainder = total - sum(values.values())

        order = sorted(
            EXPECTED_SPLITS,
            key=lambda split: (
                raw[split] - values[split],
                -EXPECTED_SPLITS.index(split),
            ),
            reverse=True,
        )

        for split in order[:remainder]:
            values[split] += 1

        if sum(values.values()) != total:
            raise RuntimeError(
                f"Category target calculation failed for {category}."
            )

        for split in EXPECTED_SPLITS:
            if values[split] <= 0:
                raise RuntimeError(
                    f"Category {category} would have zero rows in {split}."
                )

        targets[category] = values

    for split in EXPECTED_SPLITS:
        total = sum(targets[category][split] for category in targets)

        if total != EXPECTED_ROWS[split]:
            raise RuntimeError(
                f"Category targets do not sum to the required {split} total.\n"
                f"Expected: {EXPECTED_ROWS[split]:,}\n"
                f"Found:    {total:,}"
            )

    return targets


def create_split_plan(manifest_files):
    print("[5/9] Creating deterministic split plan...")

    lookup = {item["file"]: item for item in manifest_files}

    by_category = defaultdict(list)

    for item in manifest_files:
        by_category[item["category"]].append(item)

    category_targets = calculate_category_targets(manifest_files)

    allocations = {item["file"]: [] for item in manifest_files}

    for category in sorted(by_category):
        files = sorted(
            by_category[category],
            key=lambda item: (
                -item["row_count"],
                deterministic_key(item["file"]),
                item["file"],
            ),
        )

        remaining = dict(category_targets[category])

        for item in files:
            file_name = item["file"]
            file_rows = item["row_count"]

            candidates = [
                split
                for split in EXPECTED_SPLITS
                if remaining[split] >= file_rows
            ]

            if candidates:
                selected = max(
                    candidates,
                    key=lambda split: (
                        remaining[split],
                        -EXPECTED_SPLITS.index(split),
                    ),
                )

                allocations[file_name].append(
                    {
                        "split": selected,
                        "row_start": 1,
                        "row_end": file_rows,
                        "row_count": file_rows,
                        "partition_method": "whole_file",
                    }
                )

                remaining[selected] -= file_rows

            else:
                unallocated = file_rows

                split_order = sorted(
                    EXPECTED_SPLITS,
                    key=lambda split: (
                        remaining[split],
                        -EXPECTED_SPLITS.index(split),
                    ),
                    reverse=True,
                )

                row_start = 1

                for split in split_order:
                    if unallocated <= 0:
                        break

                    available = remaining[split]

                    if available <= 0:
                        continue

                    amount = min(unallocated, available)

                    row_end = row_start + amount - 1

                    allocations[file_name].append(
                        {
                            "split": split,
                            "row_start": row_start,
                            "row_end": row_end,
                            "row_count": amount,
                            "partition_method": "deterministic_row_partition",
                        }
                    )

                    remaining[split] -= amount
                    unallocated -= amount
                    row_start = row_end + 1

                if unallocated != 0:
                    raise RuntimeError(
                        f"Unable to allocate all rows from {file_name}."
                    )

        for split in EXPECTED_SPLITS:
            if remaining[split] != 0:
                raise RuntimeError(
                    f"Category allocation failed.\n"
                    f"Category: {category}\n"
                    f"Split: {split}\n"
                    f"Remaining: {remaining[split]:,}"
                )

    validate_allocation_plan(manifest_files, allocations)

    split_counts = Counter()

    for parts in allocations.values():
        for part in parts:
            split_counts[part["split"]] += part["row_count"]

    print()
    print("    Split plan created successfully.")

    for split in EXPECTED_SPLITS:
        print(f"        {split:5}: {split_counts[split]:,} rows")

    return allocations


def validate_allocation_plan(manifest_files, allocations):
    print("    Validating allocation plan...")

    lookup = {item["file"]: item for item in manifest_files}

    if set(allocations) != set(lookup):
        raise RuntimeError(
            "Allocation file set does not match source manifest."
        )

    split_counts = Counter()
    category_split_counts = defaultdict(Counter)

    partitioned_files = set()

    for file_name, parts in allocations.items():
        source_rows = lookup[file_name]["row_count"]

        parts = sorted(parts, key=lambda item: item["row_start"])

        expected_start = 1

        for part in parts:
            if part["split"] not in EXPECTED_SPLITS:
                raise RuntimeError(
                    f"Invalid split in {file_name}: {part['split']}"
                )

            if part["row_start"] != expected_start:
                raise RuntimeError(
                    f"Gap or overlap in {file_name}.\n"
                    f"Expected row start: {expected_start}\n"
                    f"Found: {part['row_start']}"
                )

            if part["row_end"] < part["row_start"]:
                raise RuntimeError(
                    f"Invalid row range in {file_name}: {part}"
                )

            calculated_count = part["row_end"] - part["row_start"] + 1

            if calculated_count != part["row_count"]:
                raise RuntimeError(
                    f"Row count mismatch in {file_name}: {part}"
                )

            split_counts[part["split"]] += part["row_count"]

            category = lookup[file_name]["category"]

            category_split_counts[category][part["split"]] += part["row_count"]

            if part["partition_method"] == "deterministic_row_partition":
                partitioned_files.add(file_name)

            expected_start = part["row_end"] + 1

        if expected_start != source_rows + 1:
            raise RuntimeError(
                f"Source rows not completely covered for {file_name}.\n"
                f"Source rows: {source_rows:,}\n"
                f"Covered through: {expected_start - 1:,}"
            )

    for split in EXPECTED_SPLITS:
        if split_counts[split] != EXPECTED_ROWS[split]:
            raise RuntimeError(
                f"Split {split} has {split_counts[split]:,} rows; "
                f"expected {EXPECTED_ROWS[split]:,}."
            )

    categories = set(item["category"] for item in manifest_files)

    for category in categories:
        for split in EXPECTED_SPLITS:
            if category_split_counts[category][split] <= 0:
                raise RuntimeError(
                    f"Category '{category}' has no rows in {split}."
                )

    print("    No row gaps: YES")
    print("    No row overlaps: YES")
    print("    Exact split totals: YES")
    print("    All categories represented in train/val/test: YES")

    whole_file_allocations = sum(
        1
        for parts in allocations.values()
        if len(parts) == 1 and parts[0]["partition_method"] == "whole_file"
    )

    print(f"    Whole-file allocations: {whole_file_allocations}")
    print(f"    Row-partitioned files: {len(partitioned_files)}")


def build_split_manifest(manifest_files, allocations):
    lookup = {item["file"]: item for item in manifest_files}

    splits = {split: [] for split in EXPECTED_SPLITS}

    for file_name, parts in allocations.items():
        source = lookup[file_name]

        for part in parts:
            splits[part["split"]].append(
                {
                    "file": file_name,
                    "source_folder": source["source_folder"],
                    "original_attack": source["original_attack"],
                    "category": source["category"],
                    "split": part["split"],
                    "row_start": part["row_start"],
                    "row_end": part["row_end"],
                    "row_count": part["row_count"],
                    "partition_method": part["partition_method"],
                    "source_sha256": source["sha256"],
                }
            )

    for split in EXPECTED_SPLITS:
        splits[split].sort(
            key=lambda item: (item["file"], item["row_start"])
        )

    return splits


def write_manifests(manifest_files, allocations):
    print("[6/9] Writing manifests...")

    split_entries = build_split_manifest(manifest_files, allocations)

    manifest = {
        "experiment": "001",
        "dataset": "CICIoT2023_Selected",
        "source_directory": str(RAW_DIR.relative_to(PROJECT_ROOT)),
        "file_count": len(manifest_files),
        "source_files": len(manifest_files),
        "total_rows": EXPECTED_TOTAL,
        "synthetic_data": False,
        "mock_data": False,
        "sampling": False,
        "label_reconstruction": True,
        "label_reconstruction_source": "parent_directory",
        "split_method": "deterministic_file_aware_row_level_split",
        "split_description": (
            "Whole-file allocation is preferred. "
            "Deterministic row-level partitioning is used where required "
            "to satisfy exact split totals and category representation."
        ),
        "random_split": False,
        "deterministic": True,
        "random_seed": RANDOM_SEED,
        "expected_rows": EXPECTED_ROWS,
        "actual_categories": sorted(
            set(item["category"] for item in manifest_files)
        ),
        "files": manifest_files,
    }

    with (RUN_DIR / "dataset_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=4)

    split_manifest = {
        "dataset": "CICIoT2023_Selected",
        "experiment": "001",
        "split_method": "deterministic_file_aware_row_level_split",
        "random_split": False,
        "deterministic": True,
        "random_seed": RANDOM_SEED,
        "total_files": EXPECTED_FILES,
        "total_rows": EXPECTED_TOTAL,
        "expected_rows": EXPECTED_ROWS,
        "splits": {},
    }

    for split in EXPECTED_SPLITS:
        entries = split_entries[split]

        split_manifest["splits"][split] = {
            "row_count": sum(item["row_count"] for item in entries),
            "partition_count": len(entries),
            "files": entries,
        }

    with (RUN_DIR / "split_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(split_manifest, handle, indent=4)

    return split_entries


def write_distributions(manifest_files, allocations):
    lookup = {item["file"]: item for item in manifest_files}

    category_distribution = defaultdict(lambda: Counter())
    attack_distribution = defaultdict(lambda: Counter())

    for file_name, parts in allocations.items():
        source = lookup[file_name]

        for part in parts:
            split = part["split"]
            rows = part["row_count"]

            category_distribution[source["category"]][split] += rows
            attack_distribution[source["original_attack"]][split] += rows

    category_output = {}

    for category in sorted(category_distribution):
        category_output[category] = {
            split: category_distribution[category][split]
            for split in EXPECTED_SPLITS
        }

        category_output[category]["total"] = sum(
            category_output[category][split] for split in EXPECTED_SPLITS
        )

    attack_output = {}

    for attack in sorted(attack_distribution):
        attack_output[attack] = {
            split: attack_distribution[attack][split]
            for split in EXPECTED_SPLITS
        }

        attack_output[attack]["total"] = sum(
            attack_output[attack][split] for split in EXPECTED_SPLITS
        )

    with (RUN_DIR / "category_distribution.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(category_output, handle, indent=4)

    with (RUN_DIR / "attack_distribution.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(attack_output, handle, indent=4)


def write_partition(source_path, output_path, row_start, row_end, metadata):
    if row_end < row_start:
        return 0

    written = 0
    current_data_row = 0

    for chunk in pd.read_csv(
        source_path, chunksize=CHUNK_SIZE, low_memory=False
    ):
        chunk_start = current_data_row + 1
        chunk_end = current_data_row + len(chunk)

        if chunk_end < row_start:
            current_data_row = chunk_end
            continue

        if chunk_start > row_end:
            break

        local_start = max(row_start, chunk_start)
        local_end = min(row_end, chunk_end)

        start_index = local_start - chunk_start
        end_index = local_end - chunk_start + 1

        selected = chunk.iloc[start_index:end_index].copy()

        selected["Original_Attack"] = metadata["original_attack"]
        selected["Category"] = metadata["category"]

        selected.to_csv(
            output_path, mode="a", header=False, index=False
        )

        written += len(selected)

        current_data_row = chunk_end

        if current_data_row >= row_end:
            break

    expected = row_end - row_start + 1

    if written != expected:
        raise RuntimeError(
            f"Partition row mismatch for {source_path}.\n"
            f"Requested: {row_start:,}-{row_end:,} ({expected:,} rows)\n"
            f"Written: {written:,}"
        )

    return written


def write_cleaned_split(split, split_entries, manifest_lookup):
    output_dir = CLEANED_DIR / split

    output_dir.mkdir(parents=True, exist_ok=True)

    combined_path = output_dir / f"{split}_combined.csv"

    if combined_path.exists():
        combined_path.unlink()

    expected_cleaned_columns = EXPECTED_COLUMNS + [
        "Original_Attack",
        "Category",
    ]

    with combined_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(expected_cleaned_columns)

    entries = [entry for entry in split_entries if entry["split"] == split]

    entries.sort(key=lambda entry: (entry["file"], entry["row_start"]))

    total_rows = 0
    processed_files = set()

    for index, entry in enumerate(entries, start=1):
        relative_path = entry["file"]

        source_path = RAW_DIR / relative_path

        metadata = manifest_lookup[relative_path]

        if relative_path not in processed_files:
            print(f"    [{index}/{len(entries)}] {relative_path}")
            processed_files.add(relative_path)

        rows_written = write_partition(
            source_path=source_path,
            output_path=combined_path,
            row_start=entry["row_start"],
            row_end=entry["row_end"],
            metadata=metadata,
        )

        total_rows += rows_written

    expected = EXPECTED_ROWS[split]

    if total_rows != expected:
        raise RuntimeError(
            f"{split} cleaned row count mismatch.\n"
            f"Expected: {expected:,}\n"
            f"Found:    {total_rows:,}"
        )

    return combined_path, total_rows


def verify_cleaned_file(path, expected_rows):
    rows = count_csv_rows(path)

    if rows != expected_rows:
        raise RuntimeError(
            f"Cleaned file verification failed for {path}.\n"
            f"Expected: {expected_rows:,}\n"
            f"Found:    {rows:,}"
        )

    header = read_header(path)

    expected_columns = EXPECTED_COLUMNS + [
        "Original_Attack",
        "Category",
    ]

    if header != expected_columns:
        raise RuntimeError(f"Cleaned schema mismatch in {path}.")

    return rows


def verify_cleaned_categories():
    print("    Verifying categories in cleaned files...")

    expected_categories = set(CATEGORY_MAP.values())

    for split in EXPECTED_SPLITS:
        path = CLEANED_DIR / split / f"{split}_combined.csv"

        found = Counter()

        for chunk in pd.read_csv(
            path, usecols=["Category"], chunksize=CHUNK_SIZE
        ):
            counts = (
                chunk["Category"].dropna().astype(str).value_counts()
            )

            found.update(counts.to_dict())

        print(f"        {split:5}: {dict(found)}")

        missing = expected_categories - set(found)

        if missing:
            raise RuntimeError(
                f"{split} is missing categories: {sorted(missing)}"
            )

        zero_categories = [
            category
            for category in expected_categories
            if found[category] <= 0
        ]

        if zero_categories:
            raise RuntimeError(
                f"{split} contains zero rows for: {sorted(zero_categories)}"
            )

    print("    Every actual category appears in train, validation and test: YES")


def verify_cleaned_totals(cleaned_results):
    total = 0

    for split in EXPECTED_SPLITS:
        expected = EXPECTED_ROWS[split]
        actual = cleaned_results[split]["rows"]

        if actual != expected:
            raise RuntimeError(
                f"Final {split} total mismatch.\n"
                f"Expected: {expected:,}\n"
                f"Found:    {actual:,}"
            )

        total += actual

    if total != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Final total mismatch.\n"
            f"Expected: {EXPECTED_TOTAL:,}\n"
            f"Found:    {total:,}"
        )

    return total


def verify_source_hashes(manifest_files):
    print("    Verifying source SHA-256 hashes...")

    for item in manifest_files:
        path = RAW_DIR / item["file"]

        actual = sha256_file(path)

        if actual != item["sha256"]:
            raise RuntimeError(
                f"SHA-256 mismatch for {item['file']}.\n"
                f"Manifest: {item['sha256']}\n"
                f"Actual:   {actual}"
            )

    print("    Source SHA-256 verification: YES")


def write_experiment_summary(cleaned_results, manifest_files, allocations):
    partition_counts = {}

    for split in EXPECTED_SPLITS:
        partition_counts[split] = sum(
            1
            for parts in allocations.values()
            for part in parts
            if part["split"] == split
        )

    whole_file_count = 0
    row_partitioned_files = []

    for file_name, parts in allocations.items():
        methods = {part["partition_method"] for part in parts}

        if methods == {"whole_file"}:
            whole_file_count += 1

        if "deterministic_row_partition" in methods:
            row_partitioned_files.append(file_name)

    summary = {
        "experiment": "001",
        "dataset": "CICIoT2023_Selected",
        "status": "SUCCESS",
        "methodology": {
            "split_method": "deterministic_file_aware_row_level_split",
            "random_split": False,
            "deterministic": True,
            "random_seed": RANDOM_SEED,
            "whole_file_allocation_preferred": True,
            "row_level_partitioning_allowed": True,
            "row_level_partitioning_reason": (
                "Exact split totals and category representation require "
                "deterministic row partitioning for selected files."
            ),
        },
        "source": {
            "raw_files": len(manifest_files),
            "raw_rows": EXPECTED_TOTAL,
            "synthetic_data": False,
            "mock_data": False,
            "sampling": False,
            "data_loss": 0,
            "duplicate_rows_added": 0,
        },
        "actual_categories": sorted(
            set(item["category"] for item in manifest_files)
        ),
        "target_rows": EXPECTED_ROWS,
        "actual_rows": {
            split: cleaned_results[split]["rows"]
            for split in EXPECTED_SPLITS
        },
        "partition_counts": partition_counts,
        "whole_file_count": whole_file_count,
        "row_level_partitioned_files": sorted(row_partitioned_files),
        "cleaned_files": cleaned_results,
    }

    with (RUN_DIR / "experiment_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=4)


def main():
    print("=" * 78)
    print("EXPERIMENT 001 - DATA CHARACTERIZATION")
    print("DETERMINISTIC FILE-AWARE + ROW-LEVEL SPLIT")
    print("=" * 78)

    if not RAW_DIR.exists():
        raise RuntimeError(
            f"Raw dataset directory does not exist:\n{RAW_DIR}"
        )

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    if CLEANED_DIR.exists():
        print("\nRemoving previous cleaned dataset...")
        shutil.rmtree(CLEANED_DIR)

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("[1/9] Discovering raw CSV files...")

    files = discover_files()

    print(f"    Discovered {len(files)} CSV files")

    validate_schemas(files)

    manifest_files = build_manifest(files)

    validate_source_dataset(manifest_files)

    allocations = create_split_plan(manifest_files)

    split_entries = write_manifests(manifest_files, allocations)

    write_distributions(manifest_files, allocations)

    verify_source_hashes(manifest_files)

    print()
    print("[7/9] Creating cleaned datasets...")

    manifest_lookup = {item["file"]: item for item in manifest_files}

    cleaned_results = {}

    for split in EXPECTED_SPLITS:
        print()
        print(
            f"    Processing {split.upper()}: "
            f"{EXPECTED_ROWS[split]:,} expected rows"
        )

        output_path, row_count = write_cleaned_split(
            split=split,
            split_entries=split_entries[split],
            manifest_lookup=manifest_lookup,
        )

        verified_rows = verify_cleaned_file(output_path, EXPECTED_ROWS[split])

        cleaned_results[split] = {
            "file": str(output_path.relative_to(PROJECT_ROOT)),
            "rows": verified_rows,
            "sha256": sha256_file(output_path),
        }

        print(f"    {split.capitalize()} complete: {verified_rows:,} rows")

    print()
    print("[8/9] Performing final dataset verification...")

    verify_cleaned_categories()

    final_total = verify_cleaned_totals(cleaned_results)

    print()

    for split in EXPECTED_SPLITS:
        print(
            f"    {split:5}: "
            f"{cleaned_results[split]['rows']:,} / "
            f"{EXPECTED_ROWS[split]:,}"
        )

    print(f"    TOTAL: {final_total:,} / {EXPECTED_TOTAL:,}")

    print()
    print("[9/9] Writing final experiment summary...")

    write_experiment_summary(
        cleaned_results=cleaned_results,
        manifest_files=manifest_files,
        allocations=allocations,
    )

    print()
    print("=" * 78)
    print("[SUCCESS] Experiment 001 Complete")
    print("=" * 78)

    print(f"Train:       {EXPECTED_ROWS['train']:,} rows")
    print(f"Validation:  {EXPECTED_ROWS['val']:,} rows")
    print(f"Test:        {EXPECTED_ROWS['test']:,} rows")
    print(f"TOTAL:       {EXPECTED_TOTAL:,} rows")
    print("Data loss:   0 rows")
    print("Synthetic:   NO")
    print("Sampling:    NO")
    print("Deterministic: YES")
    print("Class coverage: Benign / DDoS / Recon / Web")
    print("=" * 78)


if __name__ == "__main__":
    main()