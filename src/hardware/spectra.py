import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
import numpy as np
import time

# ================= CONFIG =================
FS = 2_000_000          # Sample rate (Hz)
FC = 97_000_000         # Center frequency (Hz)
GAIN = 30

FFT_SIZE = 4096
AVERAGE = 10            # Number of FFTs to average
# ==========================================

# Setup HackRF
sdr = SoapySDR.Device(dict(driver="hackrf"))
sdr.setSampleRate(SOAPY_SDR_RX, 0, FS)
sdr.setFrequency(SOAPY_SDR_RX, 0, FC)
sdr.setGain(SOAPY_SDR_RX, 0, GAIN)

rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
sdr.activateStream(rx_stream)

buffer = np.empty(FFT_SIZE, dtype=np.complex64)
window = np.hanning(FFT_SIZE)

print("Receiving IQ and computing spectrum...")

try:
    while True:
        spectrum_accum = np.zeros(FFT_SIZE, dtype=np.float64)

        for _ in range(AVERAGE):
            sr = sdr.readStream(rx_stream, [buffer], FFT_SIZE)
            if sr.ret != FFT_SIZE:
                continue

            # Apply window
            x = buffer * window

            # FFT
            X = np.fft.fftshift(np.fft.fft(x))

            # Power spectrum
            spectrum_accum += np.abs(X) ** 2

        spectrum = spectrum_accum / AVERAGE

        # Convert to dB
        spectrum_db = 10 * np.log10(spectrum + 1e-12)

        # Frequency axis
        freqs = np.fft.fftshift(
            np.fft.fftfreq(FFT_SIZE, d=1 / FS)
        ) + FC

        # Print peak frequency (example output)
        peak_idx = np.argmax(spectrum_db)
        print(
            f"Peak @ {freqs[peak_idx]/1e6:.3f} MHz | "
            f"{spectrum_db[peak_idx]:.1f} dB"
        )

        time.sleep(0.1)

except KeyboardInterrupt:
    print("Stopping...")

finally:
    sdr.deactivateStream(rx_stream)
    sdr.closeStream(rx_stream)
