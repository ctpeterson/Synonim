# Data Layout Shapes for Distributed Arrays

## Core Idea

Every distributed array is parameterized by two orthogonal concerns:

- **Distribution** — *where* data lives across locales (Block, Cyclic, BlockCyclic, etc.)
- **Shape** (layout) — *how* data is physically arranged in memory on each locale (AoS, SoA, AoSoA, Tiled)

These compose independently, so users can change one without affecting the other. This is a key advantage over Chapel, which collapses both into the distribution and locks layout to AoS.

```nim
type
  DistArray*[T; D: Distribution; S: typedesc = AoS] = object
    localData: LayoutStorage[T, S]
    dist: D
    domain: Domain
```

## Shape Types

### AoS (Array of Structures) — Default
Standard interleaved layout. Simple, good locality when accessing all fields of one element.

```
Memory: [x y z m] [x y z m] [x y z m] ...
```

### SoA (Structure of Arrays)
Each field gets its own contiguous buffer. Ideal for SIMD on CPU when kernels touch one field at a time.

```
Memory: [x x x x ...][y y y y ...][z z z z ...][m m m m ...]
```

### AoSoA[W] (Array of Structure-of-Arrays tiles)
Hybrid: SoA tiles of width W. Best layout for both SIMD and GPU coalescing. W should match the hardware vector width (e.g., 8 for AVX-256 float32, 32 for NVIDIA warps, 64 for AMD wavefronts).

```
Memory: [x*8][y*8][z*8][m*8]  [x*8][y*8][z*8][m*8]  ...
        ^^^^^^^^^ tile 0 ^^^  ^^^^^^^^^ tile 1 ^^^
```

### Tiled[B] (Cache-blocked)
For multidimensional domains (stencils, matrices). Data is stored in B×B tiles for cache locality.

### Custom Shapes
Users can define domain-specific shapes (e.g., Morton/Z-order for 2D stencils, space-filling curves for particle simulations) via a concept:

```nim
type Shape = concept S
  initStorage(typedesc[S], typedesc[T], len: int) is LayoutStorage[T, S]
  fieldAccess(LayoutStorage[T, S], field: static string, idx: int) is auto
  tileWidth(typedesc[S]) is static int  # 1 for AoS/SoA, W for AoSoA[W]
```

## Usage Examples

```nim
type Particle = object
  x, y, z: float32
  mass: float32

# Same distribution, different shapes
var a: DistArray[Particle, Block, SoA]        # CPU SIMD-friendly
var b: DistArray[Particle, Block, AoSoA[32]]  # GPU warp-coalesced
var c: DistArray[Particle, Block, AoS]         # Simple default

# Same shape, different distributions
var d: DistArray[Particle, Cyclic, SoA]
var e: DistArray[Particle, Block, SoA]
```

## Implementation Plan

### 1. Layout policy types
Zero-size compile-time-only types: `AoS`, `SoA`, `AoSoA[W]`, `Tiled[B]`.

### 2. `LayoutStorage[T, L]` via macros
Use Nim macros to introspect `T`'s fields at compile time and generate the appropriate storage type:
- **SoA**: one `seq[F]` (or aligned pointer + len) per field
- **AoSoA[W]**: `seq` of tile structs, each tile holding `array[W, F]` per field
- **AoS**: plain `seq[T]`

### 3. Aligned allocation
Guarantee SIMD alignment (e.g., 64-byte cache lines) via `posix_memalign` / `_aligned_malloc`. GPU device memory (via `gpuMalloc`) is already 256-byte aligned.

### 4. Field accessors
Compile-time-resolved accessors that map to direct memory access with no overhead:
- SoA: `storage.x[i]` → contiguous load
- AoSoA: `storage.tiles[i div W].x[i mod W]` → compiler turns div/mod into bit ops

### 5. Layout-aware iterators
Inner loops sized to exactly W iterations so the compiler can auto-vectorize without remainder loops.

### 6. GPU integration
Layout-aware host↔device transfers:
- SoA: memcpy each field buffer separately
- AoSoA: single memcpy of tile array (already coalesced)
- AoS: single memcpy (but poor coalescing on GPU)

## Advantage Over Chapel

| Aspect | Chapel | Synonim |
|--------|--------|---------|
| Layout control | None (always AoS) | AoS, SoA, AoSoA[W], Tiled[B], user-extensible |
| SIMD alignment | Implicit (hope optimizer helps) | Explicit aligned allocation, tile width = vector width |
| GPU coalescing | Relies on compiler | AoSoA with W=warp size guarantees coalescing |
| Cost | Runtime indirection through array classes | Zero-cost — layout resolved at compile time via generics |
| Extensibility | Closed set of distributions | Shape is a type parameter — users can add custom shapes |

## Build Order

1. `range/` — Chapel-like ranges and iteration
2. `domain/` — Index sets over ranges
3. `distribution/` — Inter-locale data mapping (Block, Cyclic)
4. Shape policies — Intra-locale memory organization (new `layout/` module or inside `distribution/`)
5. `DistArray` — Ties it all together with `[T, D, S]` parameters

The shape system is orthogonal to distribution, so it can be prototyped on a single locale first and plugged into the distributed layer later.
