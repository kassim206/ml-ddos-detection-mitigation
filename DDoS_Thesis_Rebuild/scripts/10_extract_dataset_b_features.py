#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import dpkt


FEATURES = [
    "packet_count",
    "duration_seconds",
    "header_length_mean",
    "ttl_mean",
    "fin_fraction",
    "syn_fraction",
    "rst_fraction",
    "psh_fraction",
    "ack_fraction",
    "packet_length_sum",
    "packet_length_min",
    "packet_length_max",
    "packet_length_mean",
    "packet_length_std",
    "packet_length_variance",
    "iat_mean",
    "iat_min",
    "iat_max",
    "iat_std",
    "packet_rate",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def unwrap_ipv4(frame):
    """
    Return dpkt.ip.IP from supported link-layer records.

    Exp010 tcpdump -i any PCAPs are expected to use Linux cooked capture,
    including Linux SLL2 on the validated host.
    """
    if isinstance(frame, dpkt.ip.IP):
        return frame

    data = getattr(frame, "data", None)
    if isinstance(data, dpkt.ip.IP):
        return data

    # Defensive second-level unwrap for supported dpkt link wrappers.
    data2 = getattr(data, "data", None)
    if isinstance(data2, dpkt.ip.IP):
        return data2

    return None


def open_pcap(path: Path):
    """
    Open classic PCAP or PCAPNG and return a dpkt reader.
    """
    f = path.open("rb")

    try:
        return f, dpkt.pcap.Reader(f), "pcap"
    except (ValueError, dpkt.dpkt.NeedData):
        f.seek(0)

    try:
        return f, dpkt.pcapng.Reader(f), "pcapng"
    except Exception:
        f.close()
        raise ValueError(f"Unsupported or unreadable capture: {path}")


def parse_packets(path: Path):
    f, reader, capture_format = open_pcap(path)

    packets = []
    total_records = 0
    decoded_link_records = 0
    ipv4_records = 0
    tcp_records = 0
    admitted_records = 0

    try:
        datalink = reader.datalink()

        for ts, buf in reader:
            total_records += 1

            try:
                if datalink == dpkt.pcap.DLT_LINUX_SLL:
                    frame = dpkt.sll.SLL(buf)
                elif datalink == getattr(dpkt.pcap, "DLT_LINUX_SLL2", 276):
                    # dpkt 1.9.8 may not expose an SLL2 class consistently.
                    # Linux SLL2 has a fixed 20-byte pseudo-header.
                    if len(buf) < 20:
                        continue

                    protocol = int.from_bytes(buf[0:2], "big")
                    if protocol != 0x0800:
                        continue

                    frame = dpkt.ip.IP(buf[20:])
                elif datalink == dpkt.pcap.DLT_EN10MB:
                    frame = dpkt.ethernet.Ethernet(buf)
                elif datalink == dpkt.pcap.DLT_RAW:
                    frame = dpkt.ip.IP(buf)
                else:
                    raise ValueError(
                        f"Unsupported datalink type {datalink}; "
                        "expected Linux SLL/SLL2, Ethernet, or raw IPv4"
                    )

                decoded_link_records += 1
                ip = unwrap_ipv4(frame)

                if ip is None:
                    continue

                ipv4_records += 1

                if ip.p != dpkt.ip.IP_PROTO_TCP:
                    continue

                tcp = ip.data
                if not isinstance(tcp, dpkt.tcp.TCP):
                    continue

                tcp_records += 1

                # Frozen Dataset-B scope: bidirectional victim HTTP/TCP port 80.
                if tcp.sport != 80 and tcp.dport != 80:
                    continue

                admitted_records += 1

                ip_header_length = int(ip.hl) * 4
                tcp_header_length = int(tcp.off) * 4
                ip_total_length = int(ip.len)

                if ip_header_length < 20:
                    raise ValueError("Invalid IPv4 header length")
                if tcp_header_length < 20:
                    raise ValueError("Invalid TCP header length")
                if ip_total_length <= 0:
                    raise ValueError("Invalid IPv4 total length")

                packets.append(
                    {
                        "timestamp": float(ts),
                        "packet_length": ip_total_length,
                        "header_length": ip_header_length + tcp_header_length,
                        "ttl": int(ip.ttl),
                        "fin": int(bool(tcp.flags & dpkt.tcp.TH_FIN)),
                        "syn": int(bool(tcp.flags & dpkt.tcp.TH_SYN)),
                        "rst": int(bool(tcp.flags & dpkt.tcp.TH_RST)),
                        "psh": int(bool(tcp.flags & dpkt.tcp.TH_PUSH)),
                        "ack": int(bool(tcp.flags & dpkt.tcp.TH_ACK)),
                    }
                )

            except (dpkt.dpkt.UnpackError, ValueError):
                raise

    finally:
        f.close()

    packets.sort(key=lambda x: x["timestamp"])

    parser_meta = {
        "capture_format": capture_format,
        "datalink": datalink,
        "total_capture_records": total_records,
        "decoded_link_records": decoded_link_records,
        "ipv4_records": ipv4_records,
        "tcp_records": tcp_records,
        "admitted_tcp_port_80_records": admitted_records,
    }

    return packets, parser_meta


def population_mean(values):
    return float(sum(values) / len(values))


def population_variance(values):
    mean = population_mean(values)
    return float(sum((x - mean) ** 2 for x in values) / len(values))


def population_std(values):
    return math.sqrt(population_variance(values))


def feature_row(window):
    n = len(window)
    if n == 0:
        raise ValueError("Cannot calculate features for empty window")

    timestamps = [p["timestamp"] for p in window]
    lengths = [p["packet_length"] for p in window]
    headers = [p["header_length"] for p in window]
    ttls = [p["ttl"] for p in window]

    duration = timestamps[-1] - timestamps[0]

    if duration < 0:
        raise ValueError("Negative window duration after timestamp ordering")

    iats = [
        timestamps[i] - timestamps[i - 1]
        for i in range(1, n)
    ]

    if any(x < 0 for x in iats):
        raise ValueError("Negative IAT after timestamp ordering")

    # Frozen candidates are 10 and 100, so IAT is always defined.
    if not iats:
        raise ValueError("Window requires at least two packets")

    packet_rate = (n / duration) if duration > 0 else None

    return {
        "packet_count": n,
        "duration_seconds": duration,
        "header_length_mean": population_mean(headers),
        "ttl_mean": population_mean(ttls),
        "fin_fraction": sum(p["fin"] for p in window) / n,
        "syn_fraction": sum(p["syn"] for p in window) / n,
        "rst_fraction": sum(p["rst"] for p in window) / n,
        "psh_fraction": sum(p["psh"] for p in window) / n,
        "ack_fraction": sum(p["ack"] for p in window) / n,
        "packet_length_sum": int(sum(lengths)),
        "packet_length_min": int(min(lengths)),
        "packet_length_max": int(max(lengths)),
        "packet_length_mean": population_mean(lengths),
        "packet_length_std": population_std(lengths),
        "packet_length_variance": population_variance(lengths),
        "iat_mean": population_mean(iats),
        "iat_min": float(min(iats)),
        "iat_max": float(max(iats)),
        "iat_std": population_std(iats),
        "packet_rate": packet_rate,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Deterministic Exp010 Dataset-B fixed-packet feature extractor"
    )
    parser.add_argument("--pcap", required=True, type=Path)
    parser.add_argument("--window-size", required=True, type=int, choices=[10, 100])
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--metadata-json", required=True, type=Path)
    args = parser.parse_args()

    if not args.pcap.is_file():
        raise FileNotFoundError(args.pcap)

    packets, parser_meta = parse_packets(args.pcap)

    complete_windows = len(packets) // args.window_size
    used_packets = complete_windows * args.window_size
    discarded_tail = len(packets) - used_packets

    rows = []

    for index in range(complete_windows):
        start = index * args.window_size
        stop = start + args.window_size
        row = feature_row(packets[start:stop])

        # Provenance fields are not model features.
        rows.append(
            {
                "window_index": index,
                **row,
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)

    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["window_index"] + FEATURES,
        )
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "experiment": "Exp010",
        "dataset": "Dataset B",
        "purpose": "feature_extraction",
        "source_pcap": str(args.pcap),
        "source_pcap_sha256": sha256_file(args.pcap),
        "window_size_packets": args.window_size,
        "window_overlap": False,
        "window_stride_packets": args.window_size,
        "cross_capture_windows": False,
        "eligible_packet_count": len(packets),
        "complete_window_count": complete_windows,
        "used_packet_count": used_packets,
        "discarded_trailing_packet_count": discarded_tail,
        "feature_count": len(FEATURES),
        "feature_order": FEATURES,
        "labels_supplied_to_extractor": False,
        "parser": parser_meta,
        "output_csv": str(args.output_csv),
        "output_csv_sha256": sha256_file(args.output_csv),
        "extractor_script": str(Path(__file__)),
        "extractor_script_sha256": sha256_file(Path(__file__)),
        "dpkt_version": dpkt.__version__,
    }

    args.metadata_json.write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
