import socket
import struct
import time
import numpy as np
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32

# ================= CONFIG =================
NODE_ID = 1
FS = 2_000_000          # 2 MHz
FC = 97_000_000         # example center freq
PACKET_SAMPLES = 362

LAPTOP_IP = "192.168.1.100"
LAPTOP_PORT = 5000

UDP_MTU = 1400          # safe UDP payload
# ==========================================

# Setup UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Setup HackRF
sdr = SoapySDR.Device(dict(driver="hackrf"))
sdr.setSampleRate(SOAPY_SDR_RX, 0, FS)
sdr.setFrequency(SOAPY_SDR_RX, 0, FC)
sdr.setGain(SOAPY_SDR_RX, 0, 30)

rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
sdr.activateStream(rx_stream)

buffer = np.empty(PACKET_SAMPLES, dtype=np.complex64)

absolute_sample_index = 0

print("Streaming started...")

try:
    while True:
        sr = sdr.readStream(rx_stream, [buffer], PACKET_SAMPLES)
        if sr.ret != PACKET_SAMPLES:
            continue

        # Packet header
        header = struct.pack(
            "!I Q I Q H",
            NODE_ID,
            absolute_sample_index,
            FS,
            FC,
            PACKET_SAMPLES
        )

        payload = buffer.tobytes()
        packet = header + payload

        sock.sendto(packet, (LAPTOP_IP, LAPTOP_PORT))

        absolute_sample_index += PACKET_SAMPLES

except KeyboardInterrupt:
    print("Stopping...")

finally:
    sdr.deactivateStream(rx_stream)
    sdr.closeStream(rx_stream)
