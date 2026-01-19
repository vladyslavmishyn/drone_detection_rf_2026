"""IQ signal generation, loading, and handling utilities."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np


def generate_signal(
    fs: float,
    duration: float,
    f0: float,
    phase: float = 0.0,
) -> np.ndarray:
    """Generate complex baseband sinusoid.

    Args:
        fs: Sample rate (Hz)
        duration: Signal duration (seconds)
        f0: Tone frequency (Hz)
        phase: Initial phase (radians)

    Returns:
        Complex64 numpy array of length int(round(fs * duration))
    """
    n = int(np.round(fs * duration))
    t = np.arange(n, dtype=np.float64) / float(fs)
    sig = np.exp(1j * (2.0 * np.pi * float(f0) * t + float(phase)))
    return sig.astype(np.complex64)


def generate_chirp(
    fs: float,
    n_samples: int,
    bandwidth: float,
    phase: float = 0.0,
) -> np.ndarray:
    """Generate a complex baseband linear FM chirp centered at 0 Hz.

    Args:
        fs: Sample rate (Hz)
        n_samples: Number of samples
        bandwidth: Sweep bandwidth (Hz), from -bw/2 to +bw/2
        phase: Initial phase (radians)

    Returns:
        Complex64 chirp signal of length n_samples
    """
    t = np.arange(n_samples, dtype=np.float64) / float(fs)
    t_end = (n_samples - 1) / float(fs)
    f0 = -0.5 * float(bandwidth)
    k = float(bandwidth) / max(t_end, 1e-12)  # Hz/s

    # phase(t) = 2π (f0 t + 0.5 k t^2) + phase0
    phi = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t) + float(phase)
    sig = np.exp(1j * phi)
    return sig.astype(np.complex64)


def load_iq_file(filepath: Path | str) -> np.ndarray:
    """Load a single .iq file as complex64 samples.
    
    Args:
        filepath: Path to .iq file
        
    Returns:
        Complex64 numpy array
    """
    return np.fromfile(filepath, dtype=np.complex64)


def parse_metadata(metadata_path: Path | str) -> dict:
    """Parse metadata.txt file to extract simulation parameters and coordinates.
    
    Args:
        metadata_path: Path to metadata.txt file
        
    Returns:
        Dictionary with keys: 'fs', 'n_samples', 'n_packets', 'receivers', 'tx_pos'
    """
    metadata = {
        'fs': None,
        'n_samples': None,
        'n_packets': None,
        'receivers': [],
        'tx_pos': None
    }
    
    with open(metadata_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
        # Extract sample rate
        fs_match = re.search(r'Sample Rate:\s*([0-9.]+)\s*Hz', content)
        if fs_match:
            metadata['fs'] = float(fs_match.group(1))
        
        # Extract samples per packet
        samples_match = re.search(r'Samples per Packet:\s*(\d+)', content)
        if samples_match:
            metadata['n_samples'] = int(samples_match.group(1))
        
        # Extract packets per file
        packets_match = re.search(r'Packets per File:\s*(\d+)', content)
        if packets_match:
            metadata['n_packets'] = int(packets_match.group(1))
        
        # Extract receiver positions - simple numeric pattern
        # Matches: RX0  50.92200000  34.80000000
        rx_pattern = r'RX(\d+)\s+([0-9.]+)\s+([0-9.]+)'
        for match in re.finditer(rx_pattern, content):
            rx_id = int(match.group(1))
            lat = float(match.group(2))
            lon = float(match.group(3))
            metadata['receivers'].append([lon, lat])  # Store as [lon, lat]
        
        # Extract transmitter position (optional) - simple numeric pattern
        tx_pattern = r'TX\s+([0-9.]+)\s+([0-9.]+)'
        tx_match = re.search(tx_pattern, content)
        if tx_match:
            tx_lat = float(tx_match.group(1))
            tx_lon = float(tx_match.group(2))
            metadata['tx_pos'] = np.array([tx_lon, tx_lat])  # [lon, lat]
    
    metadata['receivers'] = np.array(metadata['receivers'], dtype=np.float64)
    
    return metadata


def load_iq_dataset(data_dir: str | Path) -> tuple[list[np.ndarray], dict]:
    """Load all IQ files and metadata from a directory.
    
    Args:
        data_dir: Directory containing {0-n}_RX.iq files and metadata.txt
        
    Returns:
        Tuple of (rx_signals, metadata)
    """
    data_dir = Path(data_dir)
    
    # Load metadata
    metadata_path = data_dir / "metadata.txt"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    metadata = parse_metadata(metadata_path)
    
    # Load IQ files - handle both naming patterns: {id}_RX.iq or RX{id}.iq
    rx_signals = []
    
    # Try pattern: 0_RX.iq, 1_RX.iq, etc.
    iq_files = sorted(data_dir.glob("*_RX.iq"), key=lambda p: int(p.stem.split('_')[0]))
    
    # Fallback: try RX0.iq, RX1.iq, etc.
    if not iq_files:
        iq_files = sorted(data_dir.glob("RX*.iq"), key=lambda p: int(''.join(filter(str.isdigit, p.stem))))
    
    if not iq_files:
        raise FileNotFoundError(f"No .iq files found in {data_dir}")
    
    for iq_file in iq_files:
        signal = load_iq_file(iq_file)
        rx_signals.append(signal)
    
    return rx_signals, metadata


def save_iq_file(filepath: Path | str, signal: np.ndarray) -> None:
    """Save complex IQ signal to binary file.
    
    Args:
        filepath: Output path for .iq file
        signal: Complex64 numpy array
    """
    signal = np.asarray(signal, dtype=np.complex64)
    signal.tofile(filepath)


def write_metadata(
    output_path: Path | str,
    fs: float,
    n_samples: int,
    n_packets: int,
    receivers: np.ndarray,
    tx_pos: Optional[np.ndarray] = None
) -> None:
    """Write metadata.txt file with simulation parameters.
    
    Args:
        output_path: Path to output metadata.txt
        fs: Sample rate in Hz
        n_samples: Samples per packet
        n_packets: Number of packets per file
        receivers: (N, 2) array of receiver positions [lon, lat]
        tx_pos: Optional (2,) array of transmitter position [lon, lat]
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Sample Rate: {fs:.0f} Hz\n")
        f.write(f"Samples per Packet: {n_samples}\n")
        f.write(f"Packets per File: {n_packets}\n\n")
        
        f.write("Receiver Positions:\n")
        for i, (lon, lat) in enumerate(receivers):
            f.write(f"RX{i}  {lat:.8f}  {lon:.8f}\n")
        
        if tx_pos is not None:
            f.write(f"\nTransmitter Position:\n")
            f.write(f"TX  {tx_pos[1]:.8f}  {tx_pos[0]:.8f}\n")
