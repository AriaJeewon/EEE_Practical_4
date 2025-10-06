import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 128  # number of samples
max_val = 4095  # 12-bit resolution

# Sine wave
x = np.linspace(0, 2*np.pi, N, endpoint=False)
sine = ((np.sin(x) + 1) / 2 * max_val).astype(int)

# Sawtooth wave (linear ramp)
saw = np.linspace(0, max_val, N, endpoint=False).astype(int)

# Triangle wave (up then down)
triangle = np.concatenate([
    np.linspace(0, max_val, N//2, endpoint=False),
    np.linspace(max_val, 0, N//2, endpoint=False)
]).astype(int)

# Print arrays in C format
def to_c_array(name, data):
    print(f"uint32_t {name}[{N}] = {{")
    print(", ".join(map(str, data)))
    print("};\n")

print("// Basic Waveform LUTs for Task 1 Part 1")
print("// Copy these arrays into your main.c file\n")

to_c_array("Sin_LUT", sine)
to_c_array("Saw_LUT", saw)
to_c_array("Triangle_LUT", triangle)

# Plot to verify
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(sine, 'b-', linewidth=2)
plt.title('Sine Wave LUT')
plt.ylabel('Value')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(saw, 'r-', linewidth=2)
plt.title('Sawtooth Wave LUT')
plt.ylabel('Value')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(triangle, 'g-', linewidth=2)
plt.title('Triangle Wave LUT')
plt.ylabel('Value')
plt.xlabel('Sample Index')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"\nAll LUTs generated with {N} samples, range 0-{max_val}")
print("Copy the arrays above into your main.c file")
