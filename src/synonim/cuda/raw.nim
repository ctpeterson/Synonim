#[
  Synonim: https://github.com/ctpeterson/Synonim
  Source file: src/synonim/cuda/raw.nim

  Raw CUDA runtime C++ FFI bindings. Provides direct access to the CUDA
  runtime API (device management, memory, streams, events, kernel launch,
  atomics, synchronization, and math intrinsics).

  Must be compiled with `nim cpp` and requires the CUDA toolkit (nvcc).

  The binding style (importcpp from cuda_runtime.h with unified Hippo*
  type aliases) is inspired by hippo.

  Acknowledgments:
    - hippo by monofuel
      https://github.com/monofuel/hippo — MIT License

  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  MIT License

  Copyright (c) 2026 Curtis Taylor Peterson

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
]#

import std/[strformat] 
import std/[strutils]

# ---------------------------------------------------------------------------
# Compile-time CUDA toolkit detection
# ---------------------------------------------------------------------------

proc detectCudaFlags(): tuple[cflags, lflags: string] =
  ## Try nvcc, then pkg-config to find CUDA include/link flags.
  # nvcc --help doesn't have a --showme equivalent, but we can check CUDA_PATH
  let cudaPath = staticExec("bash -c 'echo ${CUDA_PATH:-${CUDA_HOME:-/usr/local/cuda}}'").strip()
  let incFlag = "-I" & cudaPath & "/include"
  let libFlag = "-L" & cudaPath & "/lib64 -lcudart"
  result = (incFlag, libFlag)

const cudaFlags = detectCudaFlags()
{.passC: cudaFlags.cflags.}
{.passL: cudaFlags.lflags.}

# ---------------------------------------------------------------------------
# Basic CUDA types
# ---------------------------------------------------------------------------

type
  cudaStream_t* {.importcpp: "cudaStream_t", header: "cuda_runtime.h".} = pointer
  cudaError_t* {.importcpp: "cudaError_t", header: "cuda_runtime.h".} = cint

type
  cudaMemcpyKind* {.size: sizeof(cint), header: "cuda_runtime.h", importcpp: "cudaMemcpyKind".} = enum
    cudaMemcpyHostToHost = 0
    cudaMemcpyHostToDevice = 1
    cudaMemcpyDeviceToHost = 2
    cudaMemcpyDeviceToDevice = 3
    cudaMemcpyDefault = 4

# ---------------------------------------------------------------------------
# Dim3 and thread/block index types
# ---------------------------------------------------------------------------

type
  Dim3* {.importcpp: "dim3", header: "cuda_runtime.h", bycopy.} = object
    x* {.importc: "x".}: uint32
    y* {.importc: "y".}: uint32
    z* {.importc: "z".}: uint32

proc newDim3*(x: uint32 = 1; y: uint32 = 1; z: uint32 = 1): Dim3 =
  result.x = x; result.y = y; result.z = z

# ---------------------------------------------------------------------------
# Device management
# ---------------------------------------------------------------------------

proc cudaGetDeviceCount*(count: ptr cint): cudaError_t
  {.header: "cuda_runtime.h", importcpp: "cudaGetDeviceCount(@)".}

proc cudaSetDevice*(device: cint): cudaError_t
  {.header: "cuda_runtime.h", importcpp: "cudaSetDevice(@)".}

proc cudaGetDevice*(device: ptr cint): cudaError_t
  {.header: "cuda_runtime.h", importcpp: "cudaGetDevice(@)".}

proc cudaDeviceReset*(): cudaError_t
  {.header: "cuda_runtime.h", importcpp: "cudaDeviceReset()".}

type
  cudaDeviceProp* {.importcpp: "cudaDeviceProp", header: "cuda_runtime.h".} = object
    name* {.importc: "name".}: array[256, char]
    totalGlobalMem* {.importc: "totalGlobalMem".}: csize_t
    sharedMemPerBlock* {.importc: "sharedMemPerBlock".}: csize_t
    warpSize* {.importc: "warpSize".}: cint
    maxThreadsPerBlock* {.importc: "maxThreadsPerBlock".}: cint
    maxThreadsDim* {.importc: "maxThreadsDim".}: array[3, cint]
    maxGridSize* {.importc: "maxGridSize".}: array[3, cint]
    multiProcessorCount* {.importc: "multiProcessorCount".}: cint
    major* {.importc: "major".}: cint
    minor* {.importc: "minor".}: cint
    deviceOverlap* {.importc: "deviceOverlap".}: cint

proc cudaGetDeviceProperties*(prop: ptr cudaDeviceProp; device: cint): cudaError_t
  {.header: "cuda_runtime.h", importcpp: "cudaGetDeviceProperties(@)".}

# ---------------------------------------------------------------------------
# Memory management
# ---------------------------------------------------------------------------

proc cudaMalloc*(`ptr`: ptr pointer; size: csize_t): cudaError_t
  {.header: "cuda_runtime.h", importcpp: "cudaMalloc(@)".}

proc cudaFree*(`ptr`: pointer): cudaError_t
  {.header: "cuda_runtime.h", importcpp: "cudaFree(@)".}

proc cudaMemcpy*(dst: pointer; src: pointer; count: csize_t;
                 kind: cudaMemcpyKind): cudaError_t
  {.header: "cuda_runtime.h", importcpp: "cudaMemcpy(@)".}

proc cudaMemcpyAsync*(dst: pointer; src: pointer; count: csize_t;
                      kind: cudaMemcpyKind; stream: cudaStream_t): cudaError_t
  {.header: "cuda_runtime.h", importcpp: "cudaMemcpyAsync(@)".}

proc cudaMemset*(devPtr: pointer; value: cint; count: csize_t): cudaError_t
  {.header: "cuda_runtime.h", importcpp: "cudaMemset(@)".}

proc cudaMemcpyToSymbol*(
  symbol: pointer; 
  src: pointer; 
  count: csize_t;
  offset: csize_t = 0;
  kind: cudaMemcpyKind = cudaMemcpyHostToDevice
): cudaError_t {.header: "cuda_runtime.h", importcpp: "cudaMemcpyToSymbol(@)".}

# Page-locked host memory
proc cudaHostAlloc*(pHost: ptr pointer; size: csize_t; flags: cuint): cudaError_t
  {.header: "cuda_runtime.h", importcpp: "cudaHostAlloc(@)".}

proc cudaFreeHost*(p: pointer): cudaError_t
  {.header: "cuda_runtime.h", importcpp: "cudaFreeHost(@)".}

# ---------------------------------------------------------------------------
# Synchronization
# ---------------------------------------------------------------------------

proc cudaDeviceSynchronize*(): cudaError_t
  {.header: "cuda_runtime.h", importcpp: "cudaDeviceSynchronize()".}

proc cudaSyncthreads*() {.importcpp: "__syncthreads()", header: "cuda_runtime.h".}

# ---------------------------------------------------------------------------
# Kernel launch
# ---------------------------------------------------------------------------

proc cudaLaunchKernel*(
  function_address: pointer; 
  gridDim: Dim3;
  blockDim: Dim3; 
  args: ptr pointer;
  sharedMem: csize_t; 
  stream: cudaStream_t
): cudaError_t {.importcpp: "cudaLaunchKernel(@)", header: "cuda_runtime.h".}

# ---------------------------------------------------------------------------
# Stream management
# ---------------------------------------------------------------------------

proc cudaStreamCreate*(stream: ptr cudaStream_t): cudaError_t
  {.header: "cuda_runtime.h", importcpp: "cudaStreamCreate(@)".}

proc cudaStreamDestroy*(stream: cudaStream_t): cudaError_t
  {.header: "cuda_runtime.h", importcpp: "cudaStreamDestroy(@)".}

proc cudaStreamSynchronize*(stream: cudaStream_t): cudaError_t
  {.header: "cuda_runtime.h", importcpp: "cudaStreamSynchronize(@)".}

proc cudaStreamWaitEvent*(stream: cudaStream_t; event: pointer;
                          flags: cuint = 0): cudaError_t
  {.header: "cuda_runtime.h", importcpp: "cudaStreamWaitEvent(@)".}

# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

type
  cudaEvent_t* {.importcpp: "cudaEvent_t", header: "cuda_runtime.h".} = pointer

proc cudaEventCreate*(event: ptr cudaEvent_t): cudaError_t
  {.header: "cuda_runtime.h", importcpp: "cudaEventCreate(@)".}

proc cudaEventCreateWithFlags*(event: ptr cudaEvent_t; flags: cuint): cudaError_t
  {.header: "cuda_runtime.h", importcpp: "cudaEventCreateWithFlags(@)".}

proc cudaEventDestroy*(event: cudaEvent_t): cudaError_t
  {.header: "cuda_runtime.h", importcpp: "cudaEventDestroy(@)".}

proc cudaEventRecord*(event: cudaEvent_t; stream: cudaStream_t = nil): cudaError_t
  {.header: "cuda_runtime.h", importcpp: "cudaEventRecord(@)".}

proc cudaEventSynchronize*(event: cudaEvent_t): cudaError_t
  {.header: "cuda_runtime.h", importcpp: "cudaEventSynchronize(@)".}

proc cudaEventQuery*(event: cudaEvent_t): cudaError_t
  {.header: "cuda_runtime.h", importcpp: "cudaEventQuery(@)".}

proc cudaEventElapsedTime*(ms: ptr cfloat; start: cudaEvent_t;
                           stop: cudaEvent_t): cudaError_t
  {.header: "cuda_runtime.h", importcpp: "cudaEventElapsedTime(@)".}

# ---------------------------------------------------------------------------
# Atomics
# ---------------------------------------------------------------------------

proc cudaAtomicAdd*(address: ptr int32; val: int32): int32
  {.header: "cuda_runtime.h", importcpp: "atomicAdd(@)".}
proc cudaAtomicAdd*(address: ptr uint32; val: uint32): uint32
  {.header: "cuda_runtime.h", importcpp: "atomicAdd(@)".}

proc cudaAtomicSub*(address: ptr int32; val: int32): int32
  {.header: "cuda_runtime.h", importcpp: "atomicSub(@)".}
proc cudaAtomicSub*(address: ptr uint32; val: uint32): uint32
  {.header: "cuda_runtime.h", importcpp: "atomicSub(@)".}

proc cudaAtomicExch*(address: ptr int32; val: int32): int32
  {.header: "cuda_runtime.h", importcpp: "atomicExch(@)".}
proc cudaAtomicExch*(address: ptr uint32; val: uint32): uint32
  {.header: "cuda_runtime.h", importcpp: "atomicExch(@)".}

proc cudaAtomicCAS*(address: ptr int32; compare: int32; val: int32): int32
  {.header: "cuda_runtime.h", importcpp: "atomicCAS(@)".}
proc cudaAtomicCAS*(address: ptr uint32; compare: uint32; val: uint32): uint32
  {.header: "cuda_runtime.h", importcpp: "atomicCAS(@)".}

proc cudaAtomicMin*(address: ptr int32; val: int32): int32
  {.header: "cuda_runtime.h", importcpp: "atomicMin(@)".}
proc cudaAtomicMin*(address: ptr uint32; val: uint32): uint32
  {.header: "cuda_runtime.h", importcpp: "atomicMin(@)".}

proc cudaAtomicMax*(address: ptr int32; val: int32): int32
  {.header: "cuda_runtime.h", importcpp: "atomicMax(@)".}
proc cudaAtomicMax*(address: ptr uint32; val: uint32): uint32
  {.header: "cuda_runtime.h", importcpp: "atomicMax(@)".}

proc cudaAtomicAnd*(address: ptr int32; val: int32): int32
  {.header: "cuda_runtime.h", importcpp: "atomicAnd(@)".}
proc cudaAtomicAnd*(address: ptr uint32; val: uint32): uint32
  {.header: "cuda_runtime.h", importcpp: "atomicAnd(@)".}

proc cudaAtomicOr*(address: ptr int32; val: int32): int32
  {.header: "cuda_runtime.h", importcpp: "atomicOr(@)".}
proc cudaAtomicOr*(address: ptr uint32; val: uint32): uint32
  {.header: "cuda_runtime.h", importcpp: "atomicOr(@)".}

proc cudaAtomicXor*(address: ptr int32; val: int32): int32
  {.header: "cuda_runtime.h", importcpp: "atomicXor(@)".}
proc cudaAtomicXor*(address: ptr uint32; val: uint32): uint32
  {.header: "cuda_runtime.h", importcpp: "atomicXor(@)".}

# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

type ConstCString* {.importc: "const char*".} = object
converter toCString*(self: ConstCString): cstring
  {.importc: "(char*)", noconv, nodecl.}

proc cudaGetErrorString*(err: cudaError_t): ConstCString
  {.header: "cuda_runtime.h", importcpp: "cudaGetErrorString(@)".}

proc cudaGetLastError*(): cudaError_t
  {.header: "cuda_runtime.h", importcpp: "cudaGetLastError()".}

# ---------------------------------------------------------------------------
# Thread/block index built-ins (only valid in device code)
# ---------------------------------------------------------------------------

let
  blockDim* {.importc, header: "cuda_runtime.h".}: Dim3
  blockIdx* {.importc, header: "cuda_runtime.h".}: Dim3
  gridDim* {.importc, header: "cuda_runtime.h".}: Dim3
  threadIdx* {.importc, header: "cuda_runtime.h".}: Dim3

# ---------------------------------------------------------------------------
# Warp intrinsics
# ---------------------------------------------------------------------------

proc shflDown*(val: cfloat; delta: cint): cfloat
  {.header: "cuda_runtime.h", importcpp: "__shfl_down_sync(0xFFFFFFFF, @)".}
proc shflDown*(val: cint; delta: cint): cint
  {.header: "cuda_runtime.h", importcpp: "__shfl_down_sync(0xFFFFFFFF, @)".}
proc shfl*(val: cfloat; srcLane: cint): cfloat
  {.header: "cuda_runtime.h", importcpp: "__shfl_sync(0xFFFFFFFF, @)".}
proc shfl*(val: cint; srcLane: cint): cint
  {.header: "cuda_runtime.h", importcpp: "__shfl_sync(0xFFFFFFFF, @)".}

const WarpSize* = 32

# ---------------------------------------------------------------------------
# Device math functions
# ---------------------------------------------------------------------------

proc expf*(x: cfloat): cfloat {.header: "cuda_runtime.h", importcpp: "expf(@)".}
proc logf*(x: cfloat): cfloat {.header: "cuda_runtime.h", importcpp: "logf(@)".}
proc sinf*(x: cfloat): cfloat {.header: "cuda_runtime.h", importcpp: "sinf(@)".}
proc cosf*(x: cfloat): cfloat {.header: "cuda_runtime.h", importcpp: "cosf(@)".}
proc sqrtf*(x: cfloat): cfloat {.header: "cuda_runtime.h", importcpp: "sqrtf(@)".}
proc powf*(base: cfloat; exp: cfloat): cfloat {.header: "cuda_runtime.h", importcpp: "powf(@)".}

proc exp*(x: cdouble): cdouble {.header: "cuda_runtime.h", importcpp: "exp(@)".}
proc log*(x: cdouble): cdouble {.header: "cuda_runtime.h", importcpp: "log(@)".}
proc sin*(x: cdouble): cdouble {.header: "cuda_runtime.h", importcpp: "sin(@)".}
proc cos*(x: cdouble): cdouble {.header: "cuda_runtime.h", importcpp: "cos(@)".}
proc sqrt*(x: cdouble): cdouble {.header: "cuda_runtime.h", importcpp: "sqrt(@)".}
proc pow*(base: cdouble; exp: cdouble): cdouble {.header: "cuda_runtime.h", importcpp: "pow(@)".}

# ---------------------------------------------------------------------------
# Half-precision conversion
# ---------------------------------------------------------------------------

proc halfToFloat*(h: uint16): cfloat
  {.header: "cuda_fp16.h",
   importcpp: "[&]{ __half_raw r; r.x = (#); return __half2float(r); }()".}

proc floatToHalf*(f: cfloat): uint16
  {.header: "cuda_fp16.h",
   importcpp: "__half_raw(__float2half(#)).x".}
