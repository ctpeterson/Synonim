#[
  Synonim: https://github.com/ctpeterson/Synonim
  Source file: src/synonim/hip/raw.nim

  Raw HIP runtime C++ FFI bindings. Provides direct access to the HIP
  runtime API (device management, memory, streams, events, kernel launch,
  atomics, synchronization, and math intrinsics).

  HIP targets both AMD (ROCm) and NVIDIA (via hipcc wrapping nvcc) GPUs.
  Must be compiled with `nim cpp` and requires hipcc in PATH.

  The binding style (importcpp from hip/hip_runtime.h with unified type
  aliases) is inspired by hippo.

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
# Compile-time HIP detection
# ---------------------------------------------------------------------------

proc detectHipPlatform(): string =
  ## Detect whether hipcc targets AMD or NVIDIA.
  let env = staticExec("bash -c 'echo $HIP_PLATFORM'").strip()
  if env.len > 0: return env
  # Check for amdclang++
  let amd = staticExec("bash -c 'which amdclang++ 2>/dev/null'").strip()
  if amd.len > 0: return "amd"
  # Check for ROCm
  let rocm = staticExec("bash -c 'test -d /opt/rocm && echo yes'").strip()
  if rocm == "yes": return "amd"
  return "nvidia"

const HipPlatform* = detectHipPlatform()

# ---------------------------------------------------------------------------
# Basic HIP types
# ---------------------------------------------------------------------------

type
  hipStream_t* {.importcpp: "hipStream_t", header: "hip/hip_runtime.h".} = pointer
  hipError_t* {.importcpp: "hipError_t", header: "hip/hip_runtime.h".} = cint

type
  hipMemcpyKind* {.size: sizeof(cint), header: "hip/hip_runtime.h",
                   importcpp: "hipMemcpyKind".} = enum
    hipMemcpyHostToHost = 0
    hipMemcpyHostToDevice = 1
    hipMemcpyDeviceToHost = 2
    hipMemcpyDeviceToDevice = 3
    hipMemcpyDefault = 4

# ---------------------------------------------------------------------------
# Dim3 and thread/block index types
# ---------------------------------------------------------------------------

type
  Dim3* {.importcpp: "dim3", header: "hip/hip_runtime.h", bycopy.} = object
    x* {.importc: "x".}: uint32
    y* {.importc: "y".}: uint32
    z* {.importc: "z".}: uint32
  BlockDim* {.importcpp: "const __HIP_Coordinates<__HIP_BlockDim>", header: "hip/hip_runtime.h".} = object
    x* {.importc: "x".}: uint32
    y* {.importc: "y".}: uint32
    z* {.importc: "z".}: uint32
  BlockIdx* {.importcpp: "const __HIP_Coordinates<__HIP_BlockIdx>", header: "hip/hip_runtime.h".} = object
    x* {.importc: "x".}: uint32
    y* {.importc: "y".}: uint32
    z* {.importc: "z".}: uint32
  GridDim* {.importcpp: "const __HIP_Coordinates<__HIP_GridDim>", header: "hip/hip_runtime.h".} = object
    x* {.importc: "x".}: uint32
    y* {.importc: "y".}: uint32
    z* {.importc: "z".}: uint32
  ThreadIdx* {.importcpp: "const __HIP_Coordinates<__HIP_ThreadIdx>", header: "hip/hip_runtime.h".} = object
    x* {.importc: "x".}: uint32
    y* {.importc: "y".}: uint32
    z* {.importc: "z".}: uint32

proc newDim3*(x: uint32 = 1; y: uint32 = 1; z: uint32 = 1): Dim3 =
  result.x = x; result.y = y; result.z = z

# ---------------------------------------------------------------------------
# Device management
# ---------------------------------------------------------------------------

proc hipGetDeviceCount*(count: ptr cint): hipError_t
  {.header: "hip/hip_runtime.h", importcpp: "hipGetDeviceCount(@)".}

proc hipSetDevice*(device: cint): hipError_t
  {.header: "hip/hip_runtime.h", importcpp: "hipSetDevice(@)".}

proc hipGetDevice*(device: ptr cint): hipError_t
  {.header: "hip/hip_runtime.h", importcpp: "hipGetDevice(@)".}

proc hipDeviceReset*(): hipError_t
  {.header: "hip/hip_runtime.h", importcpp: "hipDeviceReset()".}

type
  hipDeviceProp_t* {.importcpp: "hipDeviceProp_t", header: "hip/hip_runtime.h".} = object
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
    gcnArchName* {.importc: "gcnArchName".}: array[256, char]
    deviceOverlap* {.importc: "deviceOverlap".}: cint

proc hipGetDeviceProperties*(prop: ptr hipDeviceProp_t; device: cint): hipError_t
  {.header: "hip/hip_runtime.h", importcpp: "hipGetDeviceProperties(@)".}

# ---------------------------------------------------------------------------
# Memory management
# ---------------------------------------------------------------------------

proc hipMalloc*(`ptr`: ptr pointer; size: csize_t): hipError_t
  {.header: "hip/hip_runtime.h", importcpp: "hipMalloc(@)".}

proc hipFree*(`ptr`: pointer): hipError_t
  {.header: "hip/hip_runtime.h", importcpp: "hipFree(@)".}

proc hipMemcpy*(
  dst: pointer; 
  src: pointer; 
  sizeBytes: csize_t;
  kind: hipMemcpyKind
): hipError_t {.header: "hip/hip_runtime.h", importcpp: "hipMemcpy(@)".}

proc hipMemcpyAsync*(
  dst: pointer; 
  src: pointer; 
  sizeBytes: csize_t;
  kind: hipMemcpyKind; 
  stream: hipStream_t
): hipError_t {.header: "hip/hip_runtime.h", importcpp: "hipMemcpyAsync(@)".}

proc hipMemset*(devPtr: pointer; value: cint; count: csize_t): hipError_t
  {.header: "hip/hip_runtime.h", importcpp: "hipMemset(@)".}

proc hipMemcpyToSymbol*(
  symbol: pointer; 
  src: pointer; 
  sizeBytes: csize_t;
  offset: csize_t = 0;
  kind: hipMemcpyKind = hipMemcpyHostToDevice
): hipError_t {.header: "hip/hip_runtime.h", importcpp: "hipMemcpyToSymbol(@)".}

proc hipSymbol*[T](sym: var T): pointer
  {.header: "hip/hip_runtime.h", importcpp: "HIP_SYMBOL(@)".}

# Page-locked host memory
proc hipHostAlloc*(pHost: ptr pointer; size: csize_t; flags: cuint): hipError_t
  {.header: "hip/hip_runtime.h", importcpp: "hipHostAlloc(@)".}

proc hipHostFree*(p: pointer): hipError_t
  {.header: "hip/hip_runtime.h", importcpp: "hipHostFree(@)".}

# ---------------------------------------------------------------------------
# Synchronization
# ---------------------------------------------------------------------------

proc hipDeviceSynchronize*(): hipError_t
  {.header: "hip/hip_runtime.h", importcpp: "hipDeviceSynchronize()".}

proc hipSyncthreads*() {.importcpp: "__syncthreads()", header: "hip/hip_runtime.h".}

# ---------------------------------------------------------------------------
# Kernel launch
# ---------------------------------------------------------------------------

proc hipLaunchKernel*(
  function_address: pointer; 
  numBlocks: Dim3;
  dimBlocks: Dim3; 
  args: ptr pointer;
  sharedMemBytes: csize_t;
  stream: hipStream_t
): hipError_t {.importcpp: "hipLaunchKernel(@)", header: "hip/hip_runtime.h".}

proc hipLaunchKernelGGL*(
  function_address: proc; 
  numBlocks: Dim3;
  dimBlocks: Dim3; 
  sharedMemBytes: uint32;
  stream: hipStream_t
): hipError_t {.importcpp: "hipLaunchKernelGGL(@)", header: "hip/hip_runtime.h", varargs.}

# ---------------------------------------------------------------------------
# Stream management
# ---------------------------------------------------------------------------

proc hipStreamCreate*(stream: ptr hipStream_t): hipError_t
  {.header: "hip/hip_runtime.h", importcpp: "hipStreamCreate(@)".}

proc hipStreamDestroy*(stream: hipStream_t): hipError_t
  {.header: "hip/hip_runtime.h", importcpp: "hipStreamDestroy(@)".}

proc hipStreamSynchronize*(stream: hipStream_t): hipError_t
  {.header: "hip/hip_runtime.h", importcpp: "hipStreamSynchronize(@)".}

proc hipStreamWaitEvent*(
  stream: hipStream_t; 
  event: pointer;
  flags: cuint = 0
): hipError_t {.header: "hip/hip_runtime.h", importcpp: "hipStreamWaitEvent(@)".}

# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

type
  hipEvent_t* {.importcpp: "hipEvent_t", header: "hip/hip_runtime.h".} = pointer

proc hipEventCreate*(event: ptr hipEvent_t): hipError_t
  {.header: "hip/hip_runtime.h", importcpp: "hipEventCreate(@)".}

proc hipEventCreateWithFlags*(event: ptr hipEvent_t; flags: cuint): hipError_t
  {.header: "hip/hip_runtime.h", importcpp: "hipEventCreateWithFlags(@)".}

proc hipEventDestroy*(event: hipEvent_t): hipError_t
  {.header: "hip/hip_runtime.h", importcpp: "hipEventDestroy(@)".}

proc hipEventRecord*(event: hipEvent_t; stream: hipStream_t = nil): hipError_t
  {.header: "hip/hip_runtime.h", importcpp: "hipEventRecord(@)".}

proc hipEventSynchronize*(event: hipEvent_t): hipError_t
  {.header: "hip/hip_runtime.h", importcpp: "hipEventSynchronize(@)".}

proc hipEventQuery*(event: hipEvent_t): hipError_t
  {.header: "hip/hip_runtime.h", importcpp: "hipEventQuery(@)".}

proc hipEventElapsedTime*(
  ms: ptr cfloat; 
  start: hipEvent_t;
  stop: hipEvent_t
): hipError_t {.header: "hip/hip_runtime.h", importcpp: "hipEventElapsedTime(@)".}

# ---------------------------------------------------------------------------
# Atomics
# ---------------------------------------------------------------------------

proc hipAtomicAdd*(address: ptr int32; val: int32): int32
  {.header: "hip/hip_runtime.h", importcpp: "atomicAdd(@)".}
proc hipAtomicAdd*(address: ptr uint32; val: uint32): uint32
  {.header: "hip/hip_runtime.h", importcpp: "atomicAdd(@)".}

proc hipAtomicSub*(address: ptr int32; val: int32): int32
  {.header: "hip/hip_runtime.h", importcpp: "atomicSub(@)".}
proc hipAtomicSub*(address: ptr uint32; val: uint32): uint32
  {.header: "hip/hip_runtime.h", importcpp: "atomicSub(@)".}

proc hipAtomicExch*(address: ptr int32; val: int32): int32
  {.header: "hip/hip_runtime.h", importcpp: "atomicExch(@)".}
proc hipAtomicExch*(address: ptr uint32; val: uint32): uint32
  {.header: "hip/hip_runtime.h", importcpp: "atomicExch(@)".}

proc hipAtomicCAS*(address: ptr int32; compare: int32; val: int32): int32
  {.header: "hip/hip_runtime.h", importcpp: "atomicCAS(@)".}
proc hipAtomicCAS*(address: ptr uint32; compare: uint32; val: uint32): uint32
  {.header: "hip/hip_runtime.h", importcpp: "atomicCAS(@)".}

proc hipAtomicMin*(address: ptr int32; val: int32): int32
  {.header: "hip/hip_runtime.h", importcpp: "atomicMin(@)".}
proc hipAtomicMin*(address: ptr uint32; val: uint32): uint32
  {.header: "hip/hip_runtime.h", importcpp: "atomicMin(@)".}

proc hipAtomicMax*(address: ptr int32; val: int32): int32
  {.header: "hip/hip_runtime.h", importcpp: "atomicMax(@)".}
proc hipAtomicMax*(address: ptr uint32; val: uint32): uint32
  {.header: "hip/hip_runtime.h", importcpp: "atomicMax(@)".}

proc hipAtomicAnd*(address: ptr int32; val: int32): int32
  {.header: "hip/hip_runtime.h", importcpp: "atomicAnd(@)".}
proc hipAtomicAnd*(address: ptr uint32; val: uint32): uint32
  {.header: "hip/hip_runtime.h", importcpp: "atomicAnd(@)".}

proc hipAtomicOr*(address: ptr int32; val: int32): int32
  {.header: "hip/hip_runtime.h", importcpp: "atomicOr(@)".}
proc hipAtomicOr*(address: ptr uint32; val: uint32): uint32
  {.header: "hip/hip_runtime.h", importcpp: "atomicOr(@)".}

proc hipAtomicXor*(address: ptr int32; val: int32): int32
  {.header: "hip/hip_runtime.h", importcpp: "atomicXor(@)".}
proc hipAtomicXor*(address: ptr uint32; val: uint32): uint32
  {.header: "hip/hip_runtime.h", importcpp: "atomicXor(@)".}

# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

type ConstCString* {.importc: "const char*".} = object
converter toCString*(self: ConstCString): cstring
  {.importc: "(char*)", noconv, nodecl.}

proc hipGetErrorString*(err: hipError_t): ConstCString
  {.header: "hip/hip_runtime.h", importcpp: "hipGetErrorString(@)".}

proc hipGetLastError*(): hipError_t
  {.header: "hip/hip_runtime.h", importcpp: "hipGetLastError()".}

# ---------------------------------------------------------------------------
# Thread/block index built-ins (only valid in device code)
# ---------------------------------------------------------------------------

let
  blockDim* {.importc, header: "hip/hip_runtime.h".}: BlockDim
  blockIdx* {.importc, header: "hip/hip_runtime.h".}: BlockIdx
  gridDim* {.importc, header: "hip/hip_runtime.h".}: GridDim
  threadIdx* {.importc, header: "hip/hip_runtime.h".}: ThreadIdx

# ---------------------------------------------------------------------------
# Warp intrinsics
# ---------------------------------------------------------------------------

proc shflDown*(val: cfloat; delta: cint): cfloat
  {.header: "hip/hip_runtime.h", importcpp: "__shfl_down(@)".}
proc shflDown*(val: cint; delta: cint): cint
  {.header: "hip/hip_runtime.h", importcpp: "__shfl_down(@)".}
proc shfl*(val: cfloat; srcLane: cint): cfloat
  {.header: "hip/hip_runtime.h", importcpp: "__shfl(@)".}
proc shfl*(val: cint; srcLane: cint): cint
  {.header: "hip/hip_runtime.h", importcpp: "__shfl(@)".}

const WarpSize* {.intdefine.} = 32
  ## AMD wavefront size. Defaults to 32 (RDNA 3+).
  ## Set -d:WarpSize=64 for GCN/CDNA GPUs (e.g. MI250, MI300).

# ---------------------------------------------------------------------------
# Device math functions
# ---------------------------------------------------------------------------

proc expf*(x: cfloat): cfloat {.header: "hip/hip_runtime.h", importcpp: "expf(@)".}
proc logf*(x: cfloat): cfloat {.header: "hip/hip_runtime.h", importcpp: "logf(@)".}
proc sinf*(x: cfloat): cfloat {.header: "hip/hip_runtime.h", importcpp: "sinf(@)".}
proc cosf*(x: cfloat): cfloat {.header: "hip/hip_runtime.h", importcpp: "cosf(@)".}
proc sqrtf*(x: cfloat): cfloat {.header: "hip/hip_runtime.h", importcpp: "sqrtf(@)".}
proc powf*(base: cfloat; exp: cfloat): cfloat {.header: "hip/hip_runtime.h", importcpp: "powf(@)".}

proc exp*(x: cdouble): cdouble {.header: "hip/hip_runtime.h", importcpp: "exp(@)".}
proc log*(x: cdouble): cdouble {.header: "hip/hip_runtime.h", importcpp: "log(@)".}
proc sin*(x: cdouble): cdouble {.header: "hip/hip_runtime.h", importcpp: "sin(@)".}
proc cos*(x: cdouble): cdouble {.header: "hip/hip_runtime.h", importcpp: "cos(@)".}
proc sqrt*(x: cdouble): cdouble {.header: "hip/hip_runtime.h", importcpp: "sqrt(@)".}
proc pow*(base: cdouble; exp: cdouble): cdouble {.header: "hip/hip_runtime.h", importcpp: "pow(@)".}

# ---------------------------------------------------------------------------
# Half-precision conversion
# ---------------------------------------------------------------------------

proc halfToFloat*(h: uint16): cfloat
  {.header: "hip/hip_fp16.h",
   importcpp: "[&]{ __half_raw r; r.x = (#); return __half2float(r); }()".}

proc floatToHalf*(f: cfloat): uint16
  {.header: "hip/hip_fp16.h",
   importcpp: "__half_raw(__float2half(#)).x".}
