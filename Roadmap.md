# GPU Emulator Roadmap
## From Zero to a Single Transformer Forward Pass on CPU

> **Final Goal:** Execute a real transformer forward pass (e.g. a single layer of GPT-2 small)
> using a GPU emulator you built yourself — no CUDA hardware required.
>
> **Language recommendation:** C++ for the emulator core (authenticity, performance, learning),
> Python bindings for testing and orchestration.

---

## Overview of Projects

| # | Project | What You Build | Key Concept Learned |
|---|---------|---------------|---------------------|
| 1 | PTX Interpreter | Parse & execute PTX IR instructions | GPU ISA, SIMT model |
| 2 | Thread Hierarchy | Warps, blocks, grids, divergence | How GPU parallelism actually works |
| 3 | Memory Hierarchy | Global / shared / registers / L1 | Coalescing, bank conflicts, latency |
| 4 | Kernel Launcher | Launch CUDA-like kernels on emulator | The host/device programming model |
| 5 | Core Kernels | GEMM, softmax, layernorm, gelu | The math inside every transformer |
| 6 | Transformer Pass | Assemble kernels into a forward pass | End-to-end inference on your emulator |

Each project builds directly on the last. Do not skip ahead.

---

## Project 1 — PTX Interpreter

**What is PTX?**
PTX (Parallel Thread eXecution) is NVIDIA's virtual ISA — the "assembly language" that CUDA C
compiles down to. It is stable, documented, and designed to be portable. Your emulator will
execute PTX instructions one by one on the CPU, simulating what a real GPU would do.

**Reference:** NVIDIA PTX ISA documentation (publicly available, search "PTX ISA pdf").

---

### Milestone 1.1 — Set up the project and read PTX files

**Tasks:**
- Create a C++ project with CMake. Structure: `src/`, `include/`, `tests/`, `kernels/`.
- Write a PTX file parser that can tokenize a `.ptx` file into lines and tokens.
- Identify and store the following from a PTX file:
  - `.version` and `.target` directives
  - `.visible .entry` (kernel entry points)
  - `.param` declarations (kernel parameters)
  - `.reg` declarations (virtual register files with types: `.u32`, `.f32`, `.f64`, `.pred`, etc.)
  - Individual instruction lines

**Test:** Load a hand-written minimal PTX file (`add.ptx` that adds two integers) and print its
tokens to stdout. No execution yet.

**Deliverable:** A PTX tokenizer and basic AST (Abstract Syntax Tree) of a kernel.

---

### Milestone 1.2 — Implement a virtual register file

**Tasks:**
- Design a `RegisterFile` class that maps register names (e.g. `%r0`, `%f1`, `%p2`) to typed values.
- Support these PTX register types:
  - `.u8`, `.u16`, `.u32`, `.u64` — unsigned integers
  - `.s8`, `.s16`, `.s32`, `.s64` — signed integers
  - `.f16`, `.f32`, `.f64` — floating point
  - `.pred` — predicate (boolean) registers
- Registers are **per-thread** — this is critical. Each thread has its own private register file.
- Write unit tests: create a register file, write values, read them back, verify types.

**Key insight:** PTX uses virtual registers (unlimited count). The real GPU maps them to physical
registers during compilation. You don't need to limit register count.

**Deliverable:** `RegisterFile` class with full type support and tests.

---

### Milestone 1.3 — Implement scalar arithmetic instructions

**Tasks:**
Implement the following PTX instructions as C++ functions that operate on your RegisterFile:

- **Integer arithmetic:** `add`, `sub`, `mul`, `div`, `rem`, `mad` (multiply-add)
- **Float arithmetic:** `add.f32`, `sub.f32`, `mul.f32`, `div.f32`, `fma.f32`
- **Comparison:** `setp` (set predicate) with modifiers `.eq`, `.ne`, `.lt`, `.le`, `.gt`, `.ge`
- **Conversion:** `cvt` between integer and float types
- **Move:** `mov` (copy register to register, or load immediate)
- **Bitwise:** `and`, `or`, `xor`, `not`, `shl`, `shr`

Each instruction should:
1. Read source operands from the RegisterFile
2. Perform the operation
3. Write the result back to the destination register

**Test:** Write a PTX snippet by hand that computes `(a + b) * c - d` in floats.
Execute it through your interpreter. Verify the result matches a CPU computation.

**Deliverable:** ~20 core PTX instructions working with tests.

---

### Milestone 1.4 — Implement control flow instructions

**Tasks:**
- `bra` — unconditional branch (jump to label)
- `@%p bra` — conditional branch using a predicate register
- `ret` — return from kernel
- `bar.sync` — barrier synchronization (stub it for now, real impl comes in Project 2)
- Implement a simple program counter (`PC`) that steps through instructions
- Implement a label map: scan a kernel on load, record label → instruction index

**Test:** Write a PTX loop that sums an array of 8 integers using a counter register.
Run it and verify the result.

**Key insight:** Predicate-based branching is how GPUs avoid traditional branch instructions.
`@%p0 bra TARGET` means "branch to TARGET if predicate p0 is true."

**Deliverable:** A single-threaded PTX interpreter that can execute loops and conditionals.

---

### Milestone 1.5 — Implement memory instructions (flat model)

**Tasks:**
- Allocate a flat byte array as your emulated "device memory" (e.g. 256 MB `std::vector<uint8_t>`).
- Implement `ld.global` — load from global memory into a register
- Implement `st.global` — store from a register to global memory
- Support typed loads/stores: `.u32`, `.f32`, `.f64`, etc.
- Handle pointer arithmetic: PTX addresses are just integers (offsets into your byte array)
- Implement `cvta.to.global` — convert a generic address to a global memory address

**Test:** Allocate a buffer in your emulated memory. Write a PTX kernel that loads a value,
doubles it, and stores it back. Verify the result.

**Deliverable:** PTX interpreter with working global memory load/store.

---

### Milestone 1.6 — End-to-end single-thread kernel execution

**Tasks:**
- Write a host-side `Emulator` class that:
  - Owns the device memory buffer
  - Can `malloc` and `free` regions of it (simple bump allocator is fine)
  - Can `memcpy` data in and out of device memory
  - Can load a PTX file and launch a kernel for a **single thread** (1 block, 1 thread)
- Wire together: parse PTX → build instruction list → create register file → execute

**Test:** The classic "vector add" — allocate two input arrays and one output array in emulated
memory. Upload values. Launch your PTX `vector_add` kernel for a single element. Verify the
output.

**Deliverable:** Single-thread kernel execution working end-to-end. This is your first real
milestone — celebrate it.

---

## Project 2 — Thread Hierarchy & SIMT Execution

**What you're building:** A real GPU doesn't run one thread at a time. It runs 32 threads
simultaneously in lockstep — a **warp**. Multiple warps form a **thread block**. Multiple
blocks form a **grid**. This project implements that entire hierarchy.

---

### Milestone 2.1 — Understand and model special PTX registers

**Tasks:**
Implement the following built-in PTX registers that every thread reads to know "who am I?":

- `%tid.x`, `%tid.y`, `%tid.z` — thread index within its block
- `%ntid.x`, `%ntid.y`, `%ntid.z` — block dimensions (total threads per block)
- `%ctaid.x`, `%ctaid.y`, `%ctaid.z` — block index within the grid
- `%nctaid.x`, `%nctaid.y`, `%nctaid.z` — grid dimensions

These are read-only and set at launch time. Each thread gets different `%tid` values but the
same `%ntid`, `%ctaid`, `%nctaid` (within a block).

**Test:** Write a PTX kernel that computes its global thread ID as:
`globalId = %ctaid.x * %ntid.x + %tid.x`
and stores it to `output[globalId]`. Launch with 4 blocks of 8 threads (32 threads total).
Verify that `output` contains [0, 1, 2, ..., 31].

**Deliverable:** Special register support and multi-thread launch.

---

### Milestone 2.2 — Implement independent multi-thread execution

**Tasks:**
- Create a `Thread` struct that owns: a RegisterFile, a program counter, an active flag.
- A `ThreadBlock` owns N `Thread` objects (N = block size).
- Execute threads independently, one at a time (no real parallelism yet — just simulate it).
- Each thread executes to completion before the next starts.

**Important:** This is NOT how a real GPU works (which uses warps), but it is a correct
functional model. SIMT (warp execution) comes in the next milestone.

**Test:** Re-run the vector_add kernel but now with 64 threads over a 64-element array.
All elements should be correct.

**Deliverable:** Correct multi-thread functional execution.

---

### Milestone 2.3 — Implement warp execution (SIMT model)

This is the most intellectually important milestone in the entire project. Take your time.

**What is a warp?**
A warp is 32 threads that execute the **same instruction at the same time** (Single Instruction,
Multiple Threads). They have different register values (different thread IDs, different data)
but execute in lockstep.

**Tasks:**
- Group 32 threads into a `Warp`.
- A warp has a single program counter shared by all 32 threads.
- Each step: all 32 threads execute the *current instruction* with their own register files.
- This means: for each instruction, you loop over 32 threads and apply the instruction to each.

**Why this matters:** SIMT is what makes GPUs efficient. Understanding it viscerally — by
building it — is the payoff of this entire project.

**Deliverable:** Warp-based execution where 32 threads execute in lockstep.

---

### Milestone 2.4 — Implement warp divergence

**What is divergence?**
When threads in a warp hit a conditional branch (`@%p bra TARGET`) and some threads have
`%p = true` while others have `%p = false`, the warp **diverges**. A real GPU handles this
with an **active mask** — a 32-bit integer where each bit says whether that thread is
currently "active" (should execute the current instruction).

**Tasks:**
- Add an `activeMask` (uint32_t) to each `Warp`. Initially all bits are 1 (all active).
- When a conditional branch is encountered:
  - Compute which threads take the branch and which do not.
  - Execute the "taken" path with only the taken threads active.
  - Execute the "not taken" path with only the not-taken threads active.
  - After both paths, restore the full active mask (reconvergence).
- Instructions executed when a thread's bit is 0 in the active mask have no effect
  (writes are suppressed, memory ops are skipped).
- Implement a divergence stack to handle nested conditionals.

**Test:** Write a PTX kernel where even-numbered threads write 1.0 and odd-numbered threads
write 2.0. Verify correct results despite divergence.

**Key insight:** Divergence is *expensive* on real GPUs because both paths must be executed
serially. This is why GPU code avoids `if` statements inside kernels.

**Deliverable:** Correct divergence handling with active mask.

---

### Milestone 2.5 — Implement barrier synchronization

**Tasks:**
- Implement `bar.sync N` — all threads in the block must reach this instruction before any
  can proceed.
- In your emulator, since you execute warps one at a time, a simple barrier implementation is:
  - Track which warps have reached the barrier.
  - When all warps in a block have reached it, release them all.
- This is a simplification but functionally correct for single-block execution.

**Test:** A parallel prefix sum (scan) algorithm that uses `bar.sync` to coordinate threads.
Verify correctness.

**Deliverable:** Working barrier sync. Thread blocks now fully functional.

---

### Milestone 2.6 — Multi-block grid execution

**Tasks:**
- Create a `Grid` that owns multiple `ThreadBlock` objects.
- Blocks can execute in any order (real GPUs schedule them on streaming multiprocessors).
- For simplicity, execute blocks sequentially.
- Each block gets its own `%ctaid` values.

**Test:** A vector add over 1024 elements using 32 blocks of 32 threads each. Verify all
1024 results.

**Deliverable:** Full grid execution. Your emulator now has the complete SIMT model.

---

## Project 3 — Memory Hierarchy

**What you're building:** A real GPU has multiple levels of memory with different speeds,
sizes, and access patterns. This project adds those levels to your emulator and teaches you
why memory access patterns matter enormously for performance.

---

### Milestone 3.1 — Implement shared memory

**What is shared memory?**
Shared memory is a small, fast scratchpad (typically 48–96 KB) that all threads **within a
block** can read and write. It is much faster than global memory and is explicitly managed
by the programmer.

**Tasks:**
- Allocate a shared memory region per thread block (e.g. 48 KB).
- Implement `ld.shared` and `st.shared` instructions.
- Implement `.shared` variable declarations in PTX (e.g. `.shared .f32 sdata[256]`).
- Shared memory is zeroed at block launch.

**Test:** Implement a parallel reduction (sum of 256 elements) using shared memory.
Threads load from global into shared, sync, then reduce in shared memory.

**Deliverable:** Working shared memory with load/store.

---

### Milestone 3.2 — Implement bank conflict detection

**What are bank conflicts?**
Shared memory is divided into 32 banks (one per thread in a warp). If two threads access
the same bank simultaneously, they must serialize — a **bank conflict**. This is a major
source of performance bugs.

**Tasks:**
- Divide your shared memory into 32 banks of 4 bytes each (bank = address % 32).
- When a warp executes `ld.shared` or `st.shared`, analyze all 32 access addresses.
- Count how many threads access each bank.
- If > 1 thread accesses the same bank: log a bank conflict warning with the count
  (e.g. "2-way bank conflict at instruction X").
- Track total bank conflicts as a performance counter.

**This is emulator-only behavior** — you're not slowing down execution, just instrumenting it.

**Test:** Write a kernel with intentional bank conflicts (stride-32 access pattern) and one
without (stride-1). Verify your detector fires on the first and is silent on the second.

**Deliverable:** Bank conflict detector and performance counter.

---

### Milestone 3.3 — Implement global memory coalescing analysis

**What is coalescing?**
When a warp accesses global memory, the GPU hardware checks if all 32 accesses fall within
the same 128-byte cache line. If so, it's one memory transaction. If not, it may become
32 separate transactions — 32x slower. This is called **coalescing**.

**Tasks:**
- When a warp executes `ld.global` or `st.global`, collect all 32 addresses.
- Group them into 128-byte cache lines (address / 128).
- Count unique cache lines accessed.
- If unique cache lines == 1: coalesced (1 transaction). Log it.
- If unique cache lines > 1: uncoalesced. Log the number of transactions.
- Track total memory transactions as a performance counter.

**Test:** Implement matrix access in row-major vs column-major order. Verify that row-major
(coalesced) shows 1 transaction per warp, column-major shows 32.

**Key insight:** This is why matrix transpose is a non-trivial CUDA exercise. You'll
understand it deeply by building this.

**Deliverable:** Coalescing analyzer and transaction counter.

---

### Milestone 3.4 — Implement a simple L1/L2 cache model

**Tasks:**
- Implement an L1 cache: 32 KB, 4-way set-associative, 128-byte lines, LRU eviction.
- Implement an L2 cache: 4 MB, 8-way set-associative, 128-byte lines, LRU eviction.
- Global memory reads check L1 first, then L2, then "device memory" (your byte array).
- Track: L1 hit rate, L2 hit rate, DRAM accesses.
- Global memory writes: use write-through for simplicity (write to all levels).

**Note:** This doesn't need to be cycle-accurate. It just needs to correctly model hit/miss
behavior and give you cache statistics.

**Test:** Access a large array sequentially (should have high cache hit rate after warmup)
vs. randomly (should have low hit rate). Verify your counters reflect this.

**Deliverable:** L1/L2 cache model with hit rate statistics.

---

### Milestone 3.5 — Implement the constant memory and texture memory stubs

**Tasks:**
- Implement a constant memory region (64 KB, read-only, broadcast-optimized).
- `ld.const` — load from constant memory (same value for all threads in a warp = 1 fetch).
- Implement a minimal texture sampler stub (1D, point sampling only).
  - Texture memory is only important for some workloads; a stub is sufficient here.
- Track: constant memory broadcasts vs per-thread accesses.

**Deliverable:** Constant and texture memory stubs.

---

### Milestone 3.6 — Performance dashboard

**Tasks:**
- Build a `PerformanceCounters` struct that accumulates across a kernel launch:
  - Total instructions executed
  - Warp divergence events (and average warp efficiency)
  - Bank conflicts (total and per-kernel)
  - Memory transactions (coalesced vs uncoalesced)
  - L1/L2 cache hit rates
  - Estimated DRAM bandwidth used (bytes transferred × transactions)
- Print a formatted report after each kernel launch.

**Test:** Run your parallel reduction kernel and analyze its performance profile. Optimize it
based on the counters (e.g. fix bank conflicts if they appear). Observe counter improvements.

**Deliverable:** A full performance instrumentation dashboard. This is a major differentiator
for your project — few emulators provide this level of visibility.

---

## Project 4 — Kernel Launcher & Host/Device Interface

**What you're building:** The glue between the host (CPU code that drives your emulator) and
the device (the emulator itself). This mirrors the CUDA runtime API.

---

### Milestone 4.1 — Design the emulator API

**Tasks:**
Design a clean C++ API (and Python bindings) that mirrors CUDA's host API:

```cpp
// Device memory management
void* emu_malloc(size_t bytes);
void  emu_free(void* ptr);
void  emu_memcpy_h2d(void* dst, const void* src, size_t bytes);  // host to device
void  emu_memcpy_d2h(void* dst, const void* src, size_t bytes);  // device to host

// Kernel management
KernelHandle emu_load_ptx(const std::string& ptx_source);

// Kernel launch
void emu_launch(KernelHandle kernel,
                dim3 gridDim,    // blocks per grid
                dim3 blockDim,   // threads per block
                size_t sharedMem,
                std::vector<void*> args);  // kernel arguments (pointers into device memory)
```

**Tasks:**
- Implement all of the above.
- `dim3` is a simple struct `{x, y, z}` where z defaults to 1.
- Kernel arguments are passed as a list of device memory pointers.

**Deliverable:** Clean host API that any test can use.

---

### Milestone 4.2 — PTX argument binding

**Tasks:**
- PTX kernels declare `.param` arguments. At launch time, bind actual device memory addresses
  to these parameter names.
- When a kernel accesses a `.param` value, it gets the address you bound.
- Implement the PTX instruction `ld.param` — load a kernel parameter into a register.

**Test:** The vector_add kernel launched via `emu_launch` with 3 device memory pointers
(A, B, C) bound to its 3 parameters.

**Deliverable:** Parameter binding for kernel launches.

---

### Milestone 4.3 — Python bindings with pybind11

**Tasks:**
- Add pybind11 to your CMake project.
- Expose the emulator API to Python:
  ```python
  import gpu_emu as emu

  A_dev = emu.malloc(1024 * 4)       # 1024 floats
  emu.memcpy_h2d(A_dev, np_array)
  kernel = emu.load_ptx(open("add.ptx").read())
  emu.launch(kernel, grid=(32, 1, 1), block=(32, 1, 1), args=[A_dev, B_dev, C_dev])
  result = emu.memcpy_d2h(C_dev, 1024)
  ```
- Make kernel launch callable from NumPy arrays directly (handle the pointer extraction).

**Test:** Full vector_add end-to-end in Python. Compare result to NumPy computation.

**Deliverable:** Python-accessible emulator. This opens up much easier testing and demos.

---

### Milestone 4.4 — PTX generation from simple NumPy operations

**Tasks:**
- Write a minimal PTX code generator in Python. Given:
  - An element-wise operation description (e.g. "add two f32 arrays")
  - Array sizes
  - Thread/block configuration
- Generate valid PTX source code for it.
- This is not a general compiler — just a template-based generator for the ops you need.

**Test:** Generate PTX for `C = A * B + bias` (fused multiply-add). Execute it. Compare to
NumPy.

**Deliverable:** PTX code generator for simple array ops. Makes writing kernels much faster
for the next two projects.

---

## Project 5 — Core Transformer Kernels

**What you're building:** The actual mathematical operations used in transformer models,
implemented as PTX kernels and executed on your emulator. Each kernel corresponds to a real
component of a transformer layer.

Work with small sizes throughout: 128-dimensional model, 4-head attention, 64-token sequence
length. Correctness matters; speed is secondary.

---

### Milestone 5.1 — GEMM (General Matrix Multiply)

**Why it matters:** Matrix multiplication accounts for roughly 70–80% of all FLOPs in a
transformer. If you understand GEMM, you understand transformer compute.

**Tasks:**
- Implement a naive GEMM PTX kernel: `C = A @ B` where A is (M×K), B is (K×N), C is (M×N).
  - Each thread computes one element of C by iterating over the K dimension.
  - Launch with a 2D grid where each thread handles one output element.
- Implement a tiled GEMM using shared memory:
  - Divide A and B into tiles.
  - Load a tile of A and a tile of B into shared memory.
  - Each thread block computes a tile of C.
  - Use `bar.sync` between tile loads and computes.
- Compare performance counters between naive and tiled versions.

**Test:** Multiply two random (128×128) matrices. Compare result to NumPy's `@` operator.
Tolerance: max absolute error < 1e-5 for f32.

**Key insight:** The tiled GEMM exists to reuse data from shared memory instead of re-loading
from global memory. You'll see this in your cache counters.

**Deliverable:** Naive and tiled GEMM kernels, both verified against NumPy.

---

### Milestone 5.2 — Softmax

**Why it matters:** Softmax is applied to attention scores. It's a reduction followed by
an element-wise op — a different access pattern from GEMM.

**Tasks:**
- Implement softmax in three passes over each row:
  1. **Max reduction:** Find the maximum value in the row (for numerical stability).
  2. **Exp and sum:** Compute `exp(x - max)` for each element, accumulate the sum.
  3. **Normalize:** Divide each element by the sum.
- Use shared memory for the per-row reduction (one thread block per row).
- Use `bar.sync` between passes.

**Test:** Apply softmax to a (64×64) matrix of random floats. Compare to
`scipy.special.softmax` or a manual NumPy implementation. Tolerance: max error < 1e-5.

**Deliverable:** Numerically stable softmax kernel.

---

### Milestone 5.3 — Layer Normalization

**Why it matters:** LayerNorm normalizes each token's hidden state. Applied before or after
attention in most modern transformers (Pre-LN is now standard).

**Tasks:**
- Implement LayerNorm: for each row x of shape (d_model,):
  1. Compute mean: `μ = sum(x) / d_model`
  2. Compute variance: `σ² = sum((x - μ)²) / d_model`
  3. Normalize: `x̂ = (x - μ) / sqrt(σ² + ε)` where ε = 1e-5
  4. Scale and shift: `y = γ * x̂ + β` (γ and β are learned parameters)
- Use shared memory for mean/variance reduction.
- One thread block per row.

**Test:** Apply LayerNorm to a (64×128) matrix. Compare to `torch.nn.LayerNorm` or a NumPy
reference. Tolerance: max error < 1e-4.

**Deliverable:** LayerNorm kernel.

---

### Milestone 5.4 — GELU Activation

**Why it matters:** GELU (Gaussian Error Linear Unit) is the standard activation in the
FFN (feed-forward network) sublayer of modern transformers (used in GPT-2, BERT, etc.).

**Tasks:**
- Implement the approximate GELU formula (used in practice for speed):
  `GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
- This is fully element-wise — each thread handles one element.
- No reductions needed. Simple to parallelize.
- Implement `tanh` using PTX's `tanh.approx.f32` if available, otherwise use the exponential
  definition.

**Test:** Apply to a (64×512) tensor. Compare to `torch.nn.functional.gelu` or NumPy.
Tolerance: max error < 1e-4.

**Deliverable:** GELU kernel.

---

### Milestone 5.5 — Fused attention score computation

**Why it matters:** The attention mechanism requires: `scores = softmax(Q @ Kᵀ / sqrt(d_k))`.
This combines GEMM and softmax with a scaling factor.

**Tasks:**
- Given Q (seq×d_k), K (seq×d_k):
  1. Compute `A = Q @ Kᵀ` — use your GEMM kernel. Result: (seq×seq).
  2. Scale: `A = A / sqrt(d_k)` — element-wise divide.
  3. Apply your softmax kernel row-wise to A.
- Call your existing kernels in sequence from the Python host side.
- No masking (causal masking comes in the full pass).

**Test:** Use Q and K of shape (64×32) (64 tokens, 32-dim keys). Compare the attention
weights to a PyTorch reference. Tolerance: max error < 1e-3.

**Deliverable:** Attention score pipeline using your kernel stack.

---

### Milestone 5.6 — Causal (autoregressive) masking

**Why it matters:** Decoder-style transformers (GPT family) mask future tokens so that each
token can only attend to itself and past tokens.

**Tasks:**
- Implement a mask kernel: given an (seq×seq) attention score matrix, set
  `scores[i][j] = -inf` for all `j > i` (upper triangle).
- `-inf` in float32 is `0xFF800000` (IEEE 754). After softmax, `-inf` becomes 0.
- Add this mask application between the scaling and the softmax in Milestone 5.5.

**Test:** Verify that after softmax, the upper triangle of the attention weight matrix is
exactly 0.0.

**Deliverable:** Causal masking kernel.

---

## Project 6 — Full Transformer Forward Pass

**What you're building:** A complete single transformer decoder layer, like one of the 12
layers in GPT-2 small. Input: a sequence of token embeddings. Output: transformed embeddings.

**Architecture:** GPT-2 style decoder layer (Pre-LN variant):
1. LayerNorm
2. Multi-Head Self-Attention (Q/K/V projections → attention → output projection)
3. Residual connection
4. LayerNorm
5. Feed-Forward Network (two linear layers with GELU)
6. Residual connection

---

### Milestone 6.1 — Weight loading and embedding lookup

**Tasks:**
- Download GPT-2 small weights (124M parameters) from HuggingFace in numpy format.
  Only the first transformer layer is needed for this project.
- Write a Python loader that:
  - Loads weight tensors: `wte` (token embedding), `wpe` (position embedding),
    `c_attn.weight`, `c_attn.bias`, `c_proj.weight`, `c_proj.bias`,
    `c_fc.weight`, `c_fc.bias`, `c_proj.weight`, `c_proj.bias`, `ln_1`, `ln_2` weights.
  - Uploads each weight tensor to emulator device memory using `emu.memcpy_h2d`.
- Implement token embedding lookup:
  - Given token IDs (integers), look up rows from the embedding table.
  - Implement as a gather kernel: `output[i] = embedding_table[token_ids[i]]`.
- Implement positional embedding addition: `x = token_embed + pos_embed`.

**Test:** Encode the tokens for "Hello world" (token IDs [15496, 995]). Look up embeddings.
Verify the first 5 values of the embedding match what HuggingFace's GPT-2 produces.

**Deliverable:** Weight loader and embedding lookup working.

---

### Milestone 6.2 — Q/K/V projection

**Tasks:**
- In GPT-2, Q, K, V are computed together as one big linear projection:
  `[Q, K, V] = x @ c_attn.weight + c_attn.bias`
  where `c_attn.weight` is (d_model × 3*d_model) = (768 × 2304) for GPT-2 small.
- Use your GEMM kernel to compute this projection.
- Split the result into Q, K, V each of shape (seq × d_model).
- Reshape into multi-head format: (seq × n_heads × d_head) = (seq × 12 × 64).
- Implement the reshape/transpose as a kernel or handle it on the host with pointer arithmetic.

**Test:** Compare Q, K, V values for a 2-token input against PyTorch reference.
Tolerance: max error < 1e-3.

**Deliverable:** Q/K/V projection working for all 12 heads.

---

### Milestone 6.3 — Multi-head attention

**Tasks:**
- For each of the 12 attention heads independently:
  1. Extract head's Q slice: shape (seq × 64)
  2. Extract head's K slice: shape (seq × 64)
  3. Extract head's V slice: shape (seq × 64)
  4. Compute attention scores: `scores = Q @ Kᵀ / sqrt(64)` → shape (seq × seq)
  5. Apply causal mask
  6. Apply softmax → attention weights
  7. Compute attended values: `attended = weights @ V` → shape (seq × 64)
- Concatenate all 12 heads: result shape (seq × 768).
- You can run heads sequentially (not in parallel) — correctness first.

**Test:** Compare attention output against PyTorch reference for a 4-token input.
Tolerance: max error < 1e-2 (f32 accumulated errors across 12 heads).

**Deliverable:** Full multi-head attention working.

---

### Milestone 6.4 — Output projection and first residual connection

**Tasks:**
- Apply output projection: `attn_out = attended @ c_proj.weight + c_proj.bias`
  Shape: (seq × 768) @ (768 × 768) → (seq × 768). Use your GEMM kernel.
- Add residual: `x = x_input + attn_out` (element-wise add kernel).
- This completes the attention sublayer.

**Test:** Compare residual output against PyTorch reference. Tolerance: max error < 1e-2.

**Deliverable:** Attention sublayer complete with residual.

---

### Milestone 6.5 — Feed-Forward Network (FFN)

**Tasks:**
- Apply LayerNorm to x before the FFN.
- FFN in GPT-2 is two linear layers with GELU in between:
  1. `h = x @ c_fc.weight + c_fc.bias` → shape (seq × 3072) [4× expansion]
  2. `h = GELU(h)` [element-wise]
  3. `out = h @ c_proj.weight + c_proj.bias` → shape (seq × 768) [back to d_model]
- Use your GEMM and GELU kernels.
- Add second residual: `x = x + out`.

**Test:** Compare FFN output against PyTorch reference. Tolerance: max error < 1e-2.

**Deliverable:** FFN sublayer complete with residual.

---

### Milestone 6.6 — End-to-end single layer forward pass

**Tasks:**
- Wire everything together into a single `transformer_layer_forward()` function in Python:
  1. LayerNorm
  2. Q/K/V projection
  3. Multi-head attention with causal masking
  4. Output projection + residual
  5. LayerNorm
  6. FFN + residual
- Use real GPT-2 layer 0 weights.
- Use a real encoded prompt (e.g. "The quick brown fox").

**Final test:** Compare the full layer 0 output of your emulator against HuggingFace GPT-2's
layer 0 output for the same input. Measure max absolute error and mean absolute error.

**Success criterion:** Max absolute error < 0.05 across all sequence positions and all 768
dimensions. (Accumulated f32 rounding makes perfect equality unrealistic.)

**Deliverable:** A working single transformer layer on your GPU emulator. 🎉

---

## After Completion — How to Get Attention

**Write about it:**
- Document each project as a blog post. "I built a GPU warp divergence emulator" is a
  compelling standalone post even before the transformer pass is done.
- Post on HackerNews as "Show HN: I built a GPU emulator that runs a GPT-2 transformer layer."
- Write a deep-dive on your bank conflict detector or coalescing analyzer.

**GitHub essentials:**
- Excellent README with architecture diagrams.
- A `DESIGN.md` explaining your SIMT model and memory hierarchy decisions.
- Visualizations: print the active mask during divergence, render cache hit rates, show
  attention weights produced by your emulator.
- A demo script that takes user input text and runs it through layer 0.

**Differentiation from existing projects:**
- Your performance counters (bank conflicts, coalescing, cache hit rate) make your emulator
  *educational* in a way that gpuocelot and similar tools are not.
- The transformer-specific demo is a concrete, understandable goal that non-experts appreciate.
- The journey documentation (blog posts per milestone) builds an audience as you go.

---

## Estimated Timeline

| Project | Estimated Duration |
|---------|--------------------|
| 1 — PTX Interpreter | 4–6 weeks |
| 2 — Thread Hierarchy | 3–4 weeks |
| 3 — Memory Hierarchy | 3–4 weeks |
| 4 — Kernel Launcher | 2–3 weeks |
| 5 — Core Kernels | 4–5 weeks |
| 6 — Transformer Pass | 3–4 weeks |
| **Total** | **~5–6 months** at 10–15 hrs/week |

This assumes C++ proficiency. Add 2–4 weeks if you need to get comfortable with the language.
The timeline is aggressive but realistic if you treat it as a serious side project.

---

## Recommended Resources

| Topic | Resource |
|-------|----------|
| PTX ISA | NVIDIA PTX ISA Reference (free PDF, search "NVIDIA PTX ISA Guide") |
| GPU Architecture | "Programming Massively Parallel Processors" — Kirk & Hwu (book) |
| CUDA execution model | NVIDIA's "CUDA C++ Programming Guide" (free online) |
| Transformer architecture | "Attention Is All You Need" (Vaswani et al., 2017) — original paper |
| GPT-2 architecture | "Language Models are Unsupervised Multitask Learners" (Radford et al.) |
| GPT-2 weights + code | HuggingFace Transformers, or Andrej Karpathy's `nanoGPT` |
| Reference emulator | gpuocelot (GitHub) — academic PTX emulator to study, not copy |
| Inspiration | tinygrad (GitHub) — study how they handle kernel dispatch |
