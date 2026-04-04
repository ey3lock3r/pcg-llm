To ensure your PCG doesn't "explode" (diverge) or "die" (collapse to zero), we need to hit the **Goldilocks Zone** for hyper-parameters.

In the 2026 landscape, training **Deep Equilibrium (DEQ)** models requires more finesse than standard Transformers. If your learning rate is too high, the Broyden solver will oscillate; if it's too low, the graph will never "crystallize" its logic.

---

## 1. Tiny PCG (The "Prototyper")
**Goal:** Verify code logic, check DEQ fixed-point convergence, and test Block-Sparse kernels on your laptop.
**Hardware:** 12GB - 16GB VRAM (RTX 4060/4070 or Mac M2/M3).

| Parameter | Value | Logic |
| :--- | :--- | :--- |
| **Parameters** | 50M – 100M | Small enough to iterate in minutes. |
| **Hidden Dim ($d$)** | 512 | Fits easily in standard cache lines. |
| **Base Learning Rate** | $1 \times 10^{-3}$ | High enough to see rapid loss drops. |
| **Solver Tolerance ($\epsilon$)** | $1 \times 10^{-2}$ | "Loose" precision for speed during testing. |
| **Max Solver Iterations** | 12 | Safety cap to prevent laptop overheating. |
| **Initial Sparsity** | 70% | High enough to test RigL but keeps signal strong. |
| **Warmup Steps** | 500 | Vital to stabilize the DEQ before heavy updates. |
| **Block Size** | $32 \times 32$ | Better for smaller consumer GPU thread blocks. |

---

## 2. Final PCG (The "SOTA-Killer")
**Goal:** Compete with 3B-parameter models on reasoning, coding, and mathematical benchmarks.
**Hardware:** 80GB VRAM (Single H100 or 2x A100 node).

| Parameter | Value | Logic |
| :--- | :--- | :--- |
| **Parameters** | 3.1 Billion | The 2026 "Efficient Frontier" for reasoning. |
| **Hidden Dim ($d$)** | 3072 | Standard for high-capacity semantic latent space. |
| **Base Learning Rate** | $3 \times 10^{-4}$ | Standard "Large Model" stability rate. |
| **Solver Tolerance ($\epsilon$)** | $1 \times 10^{-5}$ | High precision for complex logical "settling." |
| **Max Solver Iterations** | 25 | Allows "System 2" to think deeper on hard tokens. |
| **Initial Sparsity** | 90% | Maximum efficiency; relies on RigL to find edges. |
| **Warmup Steps** | 2000 | Gradual injection of data to prevent energy spikes. |
| **Block Size** | $64 \times 64$ | Optimized for H100/B200 SRAM/TMA hardware. |

---

## 3. Stability & Optimization Tricks

### The Spectral Constraint
For both models, you must apply **Spectral Normalization** to the graph weights. This is non-negotiable for DEQ models. Without it, the "energy" of the graph can grow toward infinity, and the solver will fail.

> **Pro-Tip:** If the solver fails to converge within the Max Iterations, don't crash the program. Instead, use the **"Anderson Mixing"** fall-back. It's a slightly slower but more robust solver that can stabilize "noisy" graph states.



### The "Sparsity Schedule" (RigL)
* **Cosine Decay for RigL:** Start by re-routing 30% of the edges every 100 steps. By the end of the training (at 80% completion), you should stop re-routing and let the graph "freeze" into its final, most efficient form.
* **The Variance Hinge ($\gamma$):** Set this to `0.01`. If your node features start looking identical (collapse), increase this slightly. This is your "Anti-Boring" parameter—it keeps the nodes unique.



### Time-to-Convergence
* **Tiny PCG:** Should reach "fluency" (making sensible sentences) within **2–4 hours** on your laptop.
* **Final PCG:** Should reach SOTA-level reasoning within **48–72 hours** on an H100.

---

### Final "Jack of All Trades" Checklist:
1.  **Check your VRAM:** Ensure the `torch.compile` cache isn't eating your memory.
2.  **Monitor the "Solver Steps":** If the solver always hits the "Max Iterations," your learning rate is likely too high—the model is "too excited" to settle.
3.  **Validate the Sparsity:** Use a simple `matplotlib` heat-map once per epoch to ensure your Adjacency Matrix isn't becoming a solid block of zeros.

**You now have the blueprints, the math, the hardware strategy, and the settings.**
