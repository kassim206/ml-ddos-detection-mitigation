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


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
        raise ValueError("Negative window duration")

    iats = [
        timestamps[i] - timestamps[i - 1]
        for i in range(1, n)
    ]

    if not iats:
        raise ValueError("Window requires at least two packets")

    if any(x < 0 for x in iats):
        raise ValueError("Negative IAT")

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


def unwrap_ipv4(frame):
    if isinstance(frame, dpkt.ip.IP):
        return frame

    data = getattr(frame, "data", None)

    if isinstance(data, dpkt.ip.IP):
        return data

    data2 = getattr(data, "data", None)

    if isinstance(data2, dpkt.ip.IP):
        return data2

    return None


def decode_packet(ts, buf, datalink):
    if datalink == dpkt.pcap.DLT_LINUX_SLL:
        frame = dpkt.sll.SLL(buf)

    elif datalink == getattr(dpkt.pcap, "DLT_LINUX_SLL2", 276):
        if len(buf) < 20:
            return None

        protocol = int.from_bytes(buf[0:2], "big")

        if protocol != 0x0800:
            return None

        frame = dpkt.ip.IP(buf[20:])

    elif datalink == dpkt.pcap.DLT_EN10MB:
        frame = dpkt.ethernet.Ethernet(buf)

    elif datalink == dpkt.pcap.DLT_RAW:
        frame = dpkt.ip.IP(buf)

    else:
        raise ValueError(f"Unsupported datalink type {datalink}")

    ip = unwrap_ipv4(frame)

    if ip is None:
        return None

    if ip.p != dpkt.ip.IP_PROTO_TCP:
        return None

    tcp = ip.data

    if not isinstance(tcp, dpkt.tcp.TCP):
        return None

    if tcp.sport != 80 and tcp.dport != 80:
        return None

    ip_header_length = int(ip.hl) * 4
    tcp_header_length = int(tcp.off) * 4
    ip_total_length = int(ip.len)

    if ip_header_length < 20:
        raise ValueError("Invalid IPv4 header length")

    if tcp_header_length < 20:
        raise ValueError("Invalid TCP header length")

    if ip_total_length <= 0:
        raise ValueError("Invalid IPv4 total length")

    return {
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


class LiveWindowPipeline:
    def __init__(self, window_size=100):
        if window_size != 100:
            raise ValueError("Exp011 frozen window size is 100")

        self.window_size = window_size
        self.buffer = []
        self.window_index = 0
        self.rows = []
        self.admitted_packet_count = 0

    def admit(self, packet):
        if packet is None:
            return None

        self.admitted_packet_count += 1
        self.buffer.append(packet)

        if len(self.buffer) < self.window_size:
            return None

        if len(self.buffer) > self.window_size:
            raise RuntimeError("Live window buffer exceeded frozen size")

        row = {
            "window_index": self.window_index,
            **feature_row(self.buffer),
        }

        self.rows.append(row)
        self.window_index += 1
        self.buffer = []

        return row


def open_pcap(path):
    f = Path(path).open("rb")

    try:
        return f, dpkt.pcap.Reader(f), "pcap"
    except (ValueError, dpkt.dpkt.NeedData):
        f.seek(0)

    try:
        return f, dpkt.pcapng.Reader(f), "pcapng"
    except Exception:
        f.close()
        raise ValueError(f"Unsupported or unreadable capture: {path}")


def replay_pcap(path):
    f, reader, capture_format = open_pcap(path)

    pipeline = LiveWindowPipeline(100)
    total_records = 0
    decoded_errors = 0

    try:
        datalink = reader.datalink()

        for ts, buf in reader:
            total_records += 1

            try:
                packet = decode_packet(ts, buf, datalink)
            except (dpkt.dpkt.UnpackError, ValueError):
                decoded_errors += 1
                raise

            pipeline.admit(packet)

    finally:
        f.close()

    metadata = {
        "experiment": "Exp011",
        "purpose": "pre_result_live_pipeline_equivalence_validation",
        "mode": "pcap_replay",
        "source_pcap": str(path),
        "source_pcap_sha256": sha256_file(path),
        "capture_format": capture_format,
        "datalink": datalink,
        "total_capture_records": total_records,
        "admitted_tcp_port_80_records": pipeline.admitted_packet_count,
        "complete_window_count": len(pipeline.rows),
        "used_packet_count": len(pipeline.rows) * 100,
        "discarded_trailing_packet_count": len(pipeline.buffer),
        "window_size_packets": 100,
        "window_stride_packets": 100,
        "window_overlap": False,
        "cross_capture_windows": False,
        "feature_count": len(FEATURES),
        "feature_order": FEATURES,
        "labels_supplied_to_extractor": False,
        "decode_errors": decoded_errors,
        "dpkt_version": dpkt.__version__,
    }

    return pipeline.rows, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcap", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    args = parser.parse_args()

    rows, metadata = replay_pcap(args.pcap)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)

    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["window_index"] + FEATURES,
        )
        writer.writeheader()
        writer.writerows(rows)

    metadata["output_csv"] = str(args.output_csv)
    metadata["output_csv_sha256"] = sha256_file(args.output_csv)
    metadata["implementation"] = str(Path(__file__))
    metadata["implementation_sha256"] = sha256_file(Path(__file__))

    args.metadata_json.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
