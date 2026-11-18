# NeuroCraft Neural Lab: GPU Benchmarking & Policy Profiler

This module is an isolated sub-environment within the project (`neural_lab/`) dedicated to stress-testing, benchmarking, and profiling neural network architectures on the **NVIDIA GeForce RTX 5060 Laptop GPU** (8GB VRAM).

It does **not** interfere with the main Node.js/Mineflayer bot runtime.

---

## Benchmark Suite Architecture

### 1. Raw Compute & Tensor Core TFLOPS
- Measures memory transfer bandwidth (Device-to-Device copying).
- Evaluates raw GEMM (General Matrix Multiply) throughput across FP32, FP16, and BF16 precision.

### 2. Visuomotor Video Policy Simulation (ROCKET-2 / GROOT-2 / VPT)
- Simulates a 200M parameter Causal Transformer taking visual frames ($128 \times 128 \times 3$) and goal embeddings, predicting continuous mouse aiming deltas ($\Delta\theta_x, \Delta\theta_y$) and discrete keyboard chords.
- Measures single-sample real-time inference latency (P50, P95, P99) and max throughput (FPS).

### 3. High-Frequency 3D Voxel Reflex Motor Cortex
- Simulates an $11 \times 11 \times 11$ 3D Voxel spatial geometry network with entity self-attention over nearby mobs.
- Evaluates sub-tick latency (target $< 50\text{ms}$ at 20Hz) and VRAM footprint.

### 4. Small Language Model (SLM) & Reasoning Profiling
- Calculates VRAM footprint and KV-cache bounds for 1B, 3B, and 7B parameter architectures across FP16, 8-bit, and 4-bit quantizations.
- Evaluates real-time coexistence with the Minecraft client within the 8GB VRAM envelope.

### 5. VRAM Saturation Stress Test
- Progressively discovers the exact allocation ceiling before Out-Of-Memory (OOM).

---

## Running the Benchmarks

To run the suite manually:
```bat
cd neural_lab
run_benchmarks.bat
```
Results will be output to `neural_lab/benchmark_results.txt` and `neural_lab/benchmark_results.json`.
