import socket
import struct
import numpy as np
from collections import defaultdict, deque

# ================= CONFIG =================
LISTEN_IP = "0.0.0.0"
LISTEN_PORT = 5000

MAX_PACKET_SIZE = 4096

# How many samples to keep per node in memory
MAX_SAMPLES_PER_NODE = 2_000_000  # ~1 second at 2 MHz
# =========================================

# Storage per node
streams = defaultdict(dict)   # node_id -> {sample_index: complex}
sample_counts = defaultdict(int)

# UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((LISTEN_IP, LISTEN_PORT))

print("Listening for IQ packets...")

HEADER_FMT = "!I Q I Q H"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

try:
    while True:
        data, addr = sock.recvfrom(MAX_PACKET_SIZE)

        if len(data) < HEADER_SIZE:
            continue

        # Parse header
        header = data[:HEADER_SIZE]
        payload = data[HEADER_SIZE:]

        node_id, first_sample_index, fs, fc, n_samples = struct.unpack(
            HEADER_FMT, header
        )

        # Convert IQ payload
        iq = np.frombuffer(payload, dtype=np.complex64)

        if len(iq) != n_samples:
            print(f"Sample count mismatch from node {node_id}")
            continue

        # Store samples
        for i, sample in enumerate(iq):
            idx = first_sample_index + i
            streams[node_id][idx] = sample

        sample_counts[node_id] += n_samples

        # Memory control: keep only recent samples
        if len(streams[node_id]) > MAX_SAMPLES_PER_NODE:
            # Drop oldest samples
            sorted_keys = sorted(streams[node_id].keys())
            for k in sorted_keys[:len(streams[node_id]) - MAX_SAMPLES_PER_NODE]:
                del streams[node_id][k]

        print(
            f"Node {node_id} | "
            f"Received {n_samples} samples | "
            f"Total stored: {len(streams[node_id])}"
        )

except KeyboardInterrupt:
    print("Stopping receiver.")
