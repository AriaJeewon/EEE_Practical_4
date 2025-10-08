import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

# Configuration
TOTAL_SAMPLES = 20000
NUM_WAVEFORMS = 6  # 3 basic waveforms + 3 audio files
SAMPLES_PER_WAVEFORM = TOTAL_SAMPLES // NUM_WAVEFORMS  # 3333 samples each
RESOLUTION = 4095  # 12-bit DAC

print("="*80)
print("STM32 DAC Lookup Table Generator")
print("="*80)
print(f"Generating {SAMPLES_PER_WAVEFORM} samples per waveform")
print(f"Total samples: {SAMPLES_PER_WAVEFORM * NUM_WAVEFORMS}")
print("="*80)

# ========== TASK 1: Generate Basic Waveforms ==========

def generate_sine_lut(n_samples):
    """Generate sine wave LUT"""
    t = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
    sine = np.sin(t)
    # Normalize to 0-4095 range
    sine_normalized = ((sine + 1) / 2 * RESOLUTION).astype(int)
    return sine_normalized

def generate_sawtooth_lut(n_samples):
    """Generate sawtooth wave LUT"""
    sawtooth = np.linspace(0, RESOLUTION, n_samples, endpoint=False).astype(int)
    return sawtooth

def generate_triangle_lut(n_samples):
    """Generate triangle wave LUT"""
    half = n_samples // 2
    rising = np.linspace(0, RESOLUTION, half, endpoint=False)
    falling = np.linspace(RESOLUTION, 0, n_samples - half, endpoint=False)
    triangle = np.concatenate([rising, falling]).astype(int)
    return triangle

def process_wav_file(filepath, n_samples):
    """Process WAV file and create LUT"""
    try:
        # Read WAV file
        sample_rate, data = wavfile.read(filepath)
        print(f"\n✓ Processing: {filepath}")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Original samples: {len(data)}")
        print(f"  Data type: {data.dtype}")
        
        # Handle stereo by taking first channel
        if len(data.shape) > 1:
            data = data[:, 0]
            print(f"  Converted stereo to mono")
        
        # Normalize to 0-4095 range
        if data.dtype == np.int16:
            # 16-bit signed: -32768 to 32767
            normalized = ((data.astype(float) + 32768) / 65535 * RESOLUTION).astype(int)
        elif data.dtype == np.uint8:
            # 8-bit unsigned: 0 to 255
            normalized = (data.astype(float) / 255 * RESOLUTION).astype(int)
        else:
            # Assume float -1.0 to 1.0
            normalized = ((data + 1) / 2 * RESOLUTION).astype(int)
        
        # Resample to desired length
        if len(normalized) > n_samples:
            # Downsample
            indices = np.linspace(0, len(normalized) - 1, n_samples, dtype=int)
            resampled = normalized[indices]
        else:
            # Upsample using linear interpolation
            x_old = np.arange(len(normalized))
            x_new = np.linspace(0, len(normalized) - 1, n_samples)
            resampled = np.interp(x_new, x_old, normalized).astype(int)
        
        # Ensure values are in range
        resampled = np.clip(resampled, 0, RESOLUTION)
        
        print(f"  ✓ Resampled to: {len(resampled)} samples")
        print(f"  Value range: {resampled.min()} to {resampled.max()}")
        
        return resampled, sample_rate
    
    except Exception as e:
        print(f"✗ Error processing {filepath}: {e}")
        print(f"  Using placeholder sine wave instead")
        return generate_sine_lut(n_samples), 44100

print("\n" + "="*80)
print("GENERATING WAVEFORMS")
print("="*80)

# Generate basic waveforms
print("\n1. Generating Sine wave...")
sine_lut = generate_sine_lut(SAMPLES_PER_WAVEFORM)
print(f"   ✓ Generated {len(sine_lut)} samples")

print("\n2. Generating Sawtooth wave...")
sawtooth_lut = generate_sawtooth_lut(SAMPLES_PER_WAVEFORM)
print(f"   ✓ Generated {len(sawtooth_lut)} samples")

print("\n3. Generating Triangle wave...")
triangle_lut = generate_triangle_lut(SAMPLES_PER_WAVEFORM)
print(f"   ✓ Generated {len(triangle_lut)} samples")

# Process WAV files
print("\n" + "="*80)
print("PROCESSING WAV FILES")
print("="*80)

wav_files = [
    ("C:/Users/thele/Downloads/piano.wav", "piano_lut"),
    ("C:/Users/thele/Downloads/guitar.wav", "guitar_lut"),
    ("C:/Users/thele/Downloads/drum.wav", "drum_lut")
]

audio_luts = []
for filepath, name in wav_files:
    lut, sr = process_wav_file(filepath, SAMPLES_PER_WAVEFORM)
    audio_luts.append((lut, name, sr))

# Plot all waveforms
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle('Generated Lookup Tables (20,000 total samples)', fontsize=16, fontweight='bold')

# Basic waveforms
axes[0, 0].plot(sine_lut, 'b-', linewidth=0.5)
axes[0, 0].set_title('Sine Wave LUT', fontweight='bold')
axes[0, 0].set_ylabel('DAC Value (0-4095)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim([0, 4200])

axes[1, 0].plot(sawtooth_lut, 'r-', linewidth=0.5)
axes[1, 0].set_title('Sawtooth Wave LUT', fontweight='bold')
axes[1, 0].set_ylabel('DAC Value (0-4095)')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim([0, 4200])

axes[2, 0].plot(triangle_lut, 'g-', linewidth=0.5)
axes[2, 0].set_title('Triangle Wave LUT', fontweight='bold')
axes[2, 0].set_ylabel('DAC Value (0-4095)')
axes[2, 0].set_xlabel('Sample Index')
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].set_ylim([0, 4200])

# Audio waveforms
colors = ['purple', 'orange', 'brown']
for idx, (lut, name, sr) in enumerate(audio_luts):
    axes[idx, 1].plot(lut, colors[idx], linewidth=0.5)
    axes[idx, 1].set_title(f'{name.replace("_lut", "").title()} LUT (SR: {sr} Hz)', fontweight='bold')
    axes[idx, 1].set_ylabel('DAC Value (0-4095)')
    axes[idx, 1].grid(True, alpha=0.3)
    axes[idx, 1].set_ylim([0, 4200])
    if idx == 2:
        axes[idx, 1].set_xlabel('Sample Index')

plt.tight_layout()
plt.savefig('STM32_All_Waveforms.png', dpi=150, bbox_inches='tight')
print("✓ Plot saved as 'STM32_All_Waveforms.png'")

# ========== Generate C Code ==========

def format_c_array(arr, name, samples_per_line=12):
    """Format array as C code"""
    lines = [f"const uint16_t {name}[{len(arr)}] = {{"]
    
    for i in range(0, len(arr), samples_per_line):
        chunk = arr[i:i+samples_per_line]
        line = "    " + ", ".join(f"{val:4d}" for val in chunk)
        if i + samples_per_line < len(arr):
            line += ","
        lines.append(line)
    
    lines.append("};")
    return "\n".join(lines)

print("\n" + "="*80)
print("GENERATING C CODE")
print("="*80)

# Save to file for easy copying
output_filename = "STM32_LUT_Arrays.txt"
with open(output_filename, "w", encoding='utf-8') as f:
    f.write("/"+"*"*78 + "\n")
    f.write(" * STM32 DAC Lookup Tables - AUTO-GENERATED CODE\n")
    f.write(" * EEE3096S Practical 4 - Tasks 1 & 2\n")
    f.write(" *\n")
    f.write(f" * Total samples: {SAMPLES_PER_WAVEFORM * NUM_WAVEFORMS}\n")
    f.write(f" * Samples per LUT: {SAMPLES_PER_WAVEFORM}\n")
    f.write(" * DAC Resolution: 12-bit (0-4095)\n")
    f.write(" *\n")
    f.write(" * INSTRUCTIONS FOR USE:\n")
    f.write(" * 1. Copy ALL code below (from line after this comment block)\n")
    f.write(" * 2. Open your main.c file in STM32CubeIDE\n")
    f.write(" * 3. Find the section: /* USER CODE BEGIN PV */\n")
    f.write(" * 4. Paste all the code there\n")
    f.write(" * 5. Adjust F_SIGNAL based on your needs:\n")
    f.write(" *    - For musical tones (sine/sawtooth/triangle): 440 Hz\n")
    f.write(" *    - For audio playback (piano/guitar/drum): 44100 Hz\n")
    f.write(" *\n")
    f.write(" " + "*"*78 + "/\n\n")
    
    f.write("// ========== TASK 1: Basic Waveform Lookup Tables ==========\n")
    f.write("// These are single-cycle waveforms for tone generation\n\n")
    
    f.write(format_c_array(sine_lut, "sine_lut") + "\n\n")
    f.write(format_c_array(sawtooth_lut, "sawtooth_lut") + "\n\n")
    f.write(format_c_array(triangle_lut, "triangle_lut") + "\n\n")
    
    f.write("// ========== TASK 1: Audio File Lookup Tables ==========\n")
    f.write("// These contain actual audio samples from .wav files\n\n")
    
    for lut, name, sr in audio_luts:
        f.write(f"// Original sample rate: {sr} Hz\n")
        f.write(format_c_array(lut, name) + "\n\n")
    
    f.write("// ========== TASK 2: Timing Parameters ==========\n\n")
    
    f.write(f"#define NS {SAMPLES_PER_WAVEFORM}  // Number of samples in each LUT\n")
    f.write(f"#define TIM2CLK {16000000}UL  // Timer2 clock frequency in Hz (16 MHz)\n\n")
    
    f.write("// F_SIGNAL: Desired output frequency in Hz\n")
    f.write("// Valid range: 10 Hz to 2402 Hz (limited by Nyquist: TIM2CLK / (2 * NS))\n")
    f.write("// Recommended values:\n")
    f.write("//   - 440 Hz for musical tones (A4 note)\n")
    f.write("//   - 44100 Hz for audio playback (though this exceeds Nyquist limit)\n")
    f.write("#define F_SIGNAL 440  // Change this based on your waveform type\n\n")
    
    f.write("// ========== TASK 3: Timer Ticks Calculation ==========\n\n")
    f.write("// TIM2_TICKS determines how often the PWM duty cycle is updated\n")
    f.write("// Formula: TIM2_TICKS = TIM2CLK / (NS * F_SIGNAL)\n")
    f.write("// This value is loaded into the timer's auto-reload register\n")
    f.write("#define TIM2_TICKS (TIM2CLK / (NS * F_SIGNAL))\n\n")
    
    f.write("// Example calculations:\n")
    f.write(f"// For F_SIGNAL = 440 Hz:   TIM2_TICKS = {16000000 // (SAMPLES_PER_WAVEFORM * 440)}\n")
    f.write(f"// For F_SIGNAL = 1000 Hz:  TIM2_TICKS = {16000000 // (SAMPLES_PER_WAVEFORM * 1000)}\n")
    f.write(f"// For F_SIGNAL = 44100 Hz: TIM2_TICKS = {16000000 // (SAMPLES_PER_WAVEFORM * 44100)}\n\n")
    
    f.write("// ========== Additional Notes ==========\n")
    f.write("// F_SIGNAL Limit Explanation:\n")
    f.write("// Maximum F_SIGNAL is constrained by the Nyquist theorem:\n")
    f.write("//   F_max = TIM2CLK / (2 * NS)\n")
    f.write(f"//   F_max = {16000000} / (2 * {SAMPLES_PER_WAVEFORM})\n")
    f.write(f"//   F_max ≈ {16000000 // (2 * SAMPLES_PER_WAVEFORM)} Hz\n")
    f.write("//\n")
    f.write("// Above this frequency, you cannot accurately reproduce the waveform\n")
    f.write("// because you're not updating the PWM fast enough relative to the\n")
    f.write("// output frequency.\n")
    f.write("//\n")
    f.write("// For audio playback at 44.1 kHz, you may need to:\n")
    f.write("//   1. Reduce NS (fewer samples per LUT)\n")
    f.write("//   2. Increase TIM2CLK (higher timer frequency)\n")
    f.write("//   3. Accept some aliasing/distortion\n")

print(f"✓ C code saved to '{output_filename}'")
print(f"✓ File location: {os.path.abspath(output_filename)}")

# ========== Summary ==========

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total samples generated: {SAMPLES_PER_WAVEFORM * NUM_WAVEFORMS}")
print(f"Samples per waveform: {SAMPLES_PER_WAVEFORM}")
print(f"DAC Resolution: 12-bit (0-{RESOLUTION})")
print(f"\nWaveforms created:")
print(f"  1. Sine wave")
print(f"  2. Sawtooth wave")
print(f"  3. Triangle wave")
print(f"  4. Piano (from .wav)")
print(f"  5. Guitar (from .wav)")
print(f"  6. Drum (from .wav)")
print(f"\nFiles created:")
print(f"  • {output_filename} - Ready to copy into main.c")
print(f"  • STM32_All_Waveforms.png - Visual verification of all LUTs")
print(f"\nTask 2 Parameters:")
print(f"  NS = {SAMPLES_PER_WAVEFORM}")
print(f"  TIM2CLK = 16,000,000 Hz")
print(f"  F_SIGNAL range: 10 Hz to {16000000 // (2 * SAMPLES_PER_WAVEFORM)} Hz")
print(f"\nRecommended F_SIGNAL values:")
print(f"  • Basic waveforms: 440 Hz (A4 musical note)")
print(f"  • Audio playback: 44100 Hz (standard audio sample rate)")
print(f"\nExample TIM2_Ticks:")
print(f"  • 440 Hz:   {16000000 // (SAMPLES_PER_WAVEFORM * 440)} ticks")
print(f"  • 44100 Hz: {16000000 // (SAMPLES_PER_WAVEFORM * 44100)} ticks")
print("="*80)

print("\n" + "🎉"*40)
print("\n✅ ALL DONE! Next steps:")
print("\n📋 1. Open 'STM32_LUT_Arrays.txt'")
print("📋 2. Copy everything (Ctrl+A, Ctrl+C)")
print("📋 3. Paste into main.c in the '/* USER CODE BEGIN PV */' section")
print("📋 4. Adjust F_SIGNAL based on whether you're using tones or audio")
print("\n" + "🎉"*40)

# Show plot
plt.show()