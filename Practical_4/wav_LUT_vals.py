import numpy as np
import matplotlib.pyplot as plt
import wave
import struct

def process_wav_file(filename, target_samples=128):
    """
    Process a .wav file and create a LUT with target_samples values
    ranging from 0 to 4095 (12-bit resolution)
    """
    print(f"Processing {filename}...")
    
    try:
        # Open the WAV file
        with wave.open(filename, 'rb') as wav_file:
            # Get audio parameters
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate() // 44.1kHz
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            
            print(f"  Sample Rate: {sample_rate} Hz")
            print(f"  Channels: {channels}")
            print(f"  Sample Width: {sample_width} bytes")
            print(f"  Total Frames: {frames}")
            
            # Read audio data
            raw_audio = wav_file.readframes(frames)
            
            # Convert to numpy array based on sample width
            if sample_width == 1:  # 8-bit
                audio_data = np.frombuffer(raw_audio, dtype=np.uint8)
                audio_data = audio_data.astype(np.float64) - 128  # Center around 0
            elif sample_width == 2:  # 16-bit
                audio_data = np.frombuffer(raw_audio, dtype=np.int16)
                audio_data = audio_data.astype(np.float64)
            elif sample_width == 4:  # 32-bit
                audio_data = np.frombuffer(raw_audio, dtype=np.int32)
                audio_data = audio_data.astype(np.float64)
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            # Handle stereo by taking left channel only
            if channels == 2:
                audio_data = audio_data[::2]
            
            # Downsample to target_samples
            if len(audio_data) > target_samples:
                # Simple downsampling - take every nth sample
                step = len(audio_data) // target_samples
                audio_data = audio_data[::step][:target_samples]
            elif len(audio_data) < target_samples:
                # Upsample by repeating
                repeat_factor = target_samples // len(audio_data)
                remainder = target_samples % len(audio_data)
                audio_data = np.tile(audio_data, repeat_factor)
                if remainder > 0:
                    audio_data = np.concatenate([audio_data, audio_data[:remainder]])
            
            # Normalize to 0-4095 range
            # Use float64 to prevent overflow
            min_val = np.min(audio_data)
            max_val = np.max(audio_data)
            
            if max_val != min_val:
                # Normalize to 0-1 range
                normalized = (audio_data - min_val) / (max_val - min_val)
                # Scale to 0-4095 range
                lut_values = (normalized * 4095).astype(np.uint32)
            else:
                # All values are the same
                lut_values = np.full(target_samples, 2048, dtype=np.uint32)
            
            # Ensure values are within range
            lut_values = np.clip(lut_values, 0, 4095)
            
            print(f"  [OK] Generated LUT with {len(lut_values)} samples")
            print(f"  Original range: {min_val:.2f} to {max_val:.2f}")
            print(f"  LUT range: {np.min(lut_values)} to {np.max(lut_values)}")
            
            return lut_values
            
    except Exception as e:
        print(f"  [ERROR] Failed to process {filename}: {e}")
        return None

def to_c_array(name, data):
    """Convert numpy array to C array format"""
    print(f"uint32_t {name}[{len(data)}] = {{")
    # Print 8 values per line for readability
    for i in range(0, len(data), 8):
        line_values = data[i:i+8]
        print("  " + ", ".join(map(str, line_values)) + ("," if i+8 < len(data) else ""))
    print("};\n")

# Process all WAV files
wav_files = ['piano.wav', 'guitar.wav', 'drum.wav']
luts = {}

print("=== Audio Waveform LUT Generation for Task 1 Part 2 ===\n")

for wav_file in wav_files:
    lut = process_wav_file(wav_file)
    if lut is not None:
        luts[wav_file.replace('.wav', '')] = lut

print("\n=== Generated LUTs ===\n")

for name, lut in luts.items():
    to_c_array(f"{name.capitalize()}_LUT", lut)

# Plot all LUTs
if luts:
    plt.figure(figsize=(12, 10))
    
    for i, (name, lut) in enumerate(luts.items(), 1):
        plt.subplot(len(luts), 1, i)
        plt.plot(lut, linewidth=2)
        plt.title(f'{name.capitalize()} Wave LUT')
        plt.ylabel('Value')
        plt.xlabel('Sample Index')
        plt.grid(True)
        plt.ylim(0, 4095)
    
    plt.tight_layout()
    plt.show()

print("All audio LUTs generated successfully!")
