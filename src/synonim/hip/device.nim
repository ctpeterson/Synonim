#[
  Synonim: https://github.com/ctpeterson/Synonim
  Source file: src/synonim/hip/device.nim

  Unified GPU device abstraction layer. Provides a backend-agnostic API
  for GPU device management, memory allocation, data transfer, streams,
  events, kernel launch, and kernel attribute pragmas.

  Backend selection is controlled at compile time:
    -d:GPU=NONE     CPU-only stub (no GPU required) [default]
    -d:GPU=HIP      HIP runtime via hipcc (AMD + NVIDIA)
    -d:GPU=HIP_CPU  HIP-CPU emulation via g++ + TBB (no GPU needed)
    -d:GPU=CUDA     CUDA runtime via nvcc (NVIDIA only)

  Must be compiled with `nim cpp` for HIP, HIP_CPU, or CUDA backends.

  The design of a unified API over multiple GPU backends (with compile-time
  selection, GpuRef/GpuMemory RAII wrappers, kernel attribute macros, and
  error-to-exception conversion) is inspired by hippo.

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

import std/[macros, strformat]

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

const GPU* {.strdefine.} = "NONE"
const isHipBackend* = GPU == "HIP" or GPU == "HIP_CPU"

when isHipBackend:
  import raw as backend
  export backend
  when GPU == "HIP_CPU":
    # HIP-CPU: vendored header-only HIP emulation on CPU via Intel TBB
    # https://github.com/ROCm/HIP-CPU (MIT License, see vendor/hip-cpu/LICENSE)
    import std/os
    const hipCpuInclude = parentDir(currentSourcePath()) / "vendor" / "hip-cpu" / "include"
    {.passC: "-I" & hipCpuInclude & " -std=c++17 -D__HIP_CPU_RT__".}
    {.passL: "-ltbb -lpthread".}
elif GPU == "CUDA":
  import ../cuda/raw as backend
  export backend
elif GPU == "NONE":
  discard
else:
  {.error: "Unknown GPU backend: " & GPU &
           ". Use HIP, HIP_CPU, CUDA, or NONE.".}

when GPU != "NONE" and not defined(nimdoc):
  when defined(c) or defined(js):
    {.error: "GPU backends require `nim cpp`. Compile with `nim cpp` instead of `nim c`.".}

# ---------------------------------------------------------------------------
# Unified type aliases
# ---------------------------------------------------------------------------

when isHipBackend:
  type
    GpuStream* = hipStream_t
    GpuError* = hipError_t
    GpuEvent* = hipEvent_t
    GpuDeviceProp* = hipDeviceProp_t
  type GpuMemcpyKind* = hipMemcpyKind
  const
    gpuMemcpyHostToHost* = hipMemcpyHostToHost
    gpuMemcpyHostToDevice* = hipMemcpyHostToDevice
    gpuMemcpyDeviceToHost* = hipMemcpyDeviceToHost
    gpuMemcpyDeviceToDevice* = hipMemcpyDeviceToDevice
    gpuMemcpyDefault* = hipMemcpyDefault
elif GPU == "CUDA":
  type
    GpuStream* = cudaStream_t
    GpuError* = cudaError_t
    GpuEvent* = cudaEvent_t
    GpuDeviceProp* = cudaDeviceProp
  type GpuMemcpyKind* = cudaMemcpyKind
  const
    gpuMemcpyHostToHost* = cudaMemcpyHostToHost
    gpuMemcpyHostToDevice* = cudaMemcpyHostToDevice
    gpuMemcpyDeviceToHost* = cudaMemcpyDeviceToHost
    gpuMemcpyDeviceToDevice* = cudaMemcpyDeviceToDevice
    gpuMemcpyDefault* = cudaMemcpyDefault
else:
  # NONE backend stubs
  type
    GpuStream* = pointer
    GpuError* = cint
    GpuEvent* = pointer
    GpuDeviceProp* = object
      name*: array[256, char]
      totalGlobalMem*: csize_t
      multiProcessorCount*: cint
      warpSize*: cint
      maxThreadsPerBlock*: cint
      major*: cint
      minor*: cint
      deviceOverlap*: cint
  type GpuMemcpyKind* = enum
    gpuMemcpyHostToHost = 0
    gpuMemcpyHostToDevice = 1
    gpuMemcpyDeviceToHost = 2
    gpuMemcpyDeviceToDevice = 3
    gpuMemcpyDefault = 4
  type Dim3* = object
    x*, y*, z*: uint32

  proc newDim3*(x: uint32 = 1; y: uint32 = 1; z: uint32 = 1): Dim3 =
    Dim3(x: x, y: y, z: z)

# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

type GpuException* = object of CatchableError
  code*: cint

when GPU != "NONE":
  proc checkGpu*(err: GpuError) =
    ## Raise a GpuException if the runtime return code indicates failure.
    if err.cint != 0:
      when isHipBackend:
        let msg = $hipGetErrorString(err)
      elif GPU == "CUDA":
        let msg = $cudaGetErrorString(err)
      else:
        let msg = "GPU error code " & $err.cint
      raise (ref GpuException)(msg: msg, code: err.cint)

  template handleGpuError*(call: untyped) =
    ## Execute a GPU runtime call and raise on error.
    checkGpu(call)
else:
  proc checkGpu*(err: cint) = discard
  template handleGpuError*(call: untyped) = discard

# ---------------------------------------------------------------------------
# GpuRef — RAII wrapper for device memory
# ---------------------------------------------------------------------------

type
  GpuMemory* = object
    p*: pointer
    size*: int
  GpuRef* = ref GpuMemory

when GPU != "NONE":
  proc `=destroy`*(mem: GpuMemory) =
    if mem.p != nil:
      when isHipBackend:
        discard hipFree(mem.p)
      elif GPU == "CUDA":
        discard cudaFree(mem.p)

# ---------------------------------------------------------------------------
# Device management
# ---------------------------------------------------------------------------

when GPU != "NONE":
  proc gpuDeviceCount*(): int =
    var count: cint = 0
    when isHipBackend:
      handleGpuError hipGetDeviceCount(addr count)
    elif GPU == "CUDA":
      handleGpuError cudaGetDeviceCount(addr count)
    count.int

  proc gpuSetDevice*(device: int) =
    when isHipBackend:
      handleGpuError hipSetDevice(device.cint)
    elif GPU == "CUDA":
      handleGpuError cudaSetDevice(device.cint)

  proc gpuGetDevice*(): int =
    var device: cint = 0
    when isHipBackend:
      handleGpuError hipGetDevice(addr device)
    elif GPU == "CUDA":
      handleGpuError cudaGetDevice(addr device)
    device.int

  proc gpuDeviceReset*() =
    when isHipBackend:
      handleGpuError hipDeviceReset()
    elif GPU == "CUDA":
      handleGpuError cudaDeviceReset()

  proc gpuGetDeviceProperties*(device: int): GpuDeviceProp =
    when isHipBackend:
      handleGpuError hipGetDeviceProperties(addr result, device.cint)
    elif GPU == "CUDA":
      handleGpuError cudaGetDeviceProperties(addr result, device.cint)

  proc gpuDeviceName*(prop: GpuDeviceProp): string =
    ## Extract device name string from property struct.
    result = ""
    for c in prop.name:
      if c == '\0': break
      result.add(c)
else:
  proc gpuDeviceCount*(): int = 0
  proc gpuSetDevice*(device: int) = discard
  proc gpuGetDevice*(): int = 0
  proc gpuDeviceReset*() = discard
  proc gpuGetDeviceProperties*(device: int): GpuDeviceProp = discard
  proc gpuDeviceName*(prop: GpuDeviceProp): string = "none"

# ---------------------------------------------------------------------------
# Memory allocation and transfer
# ---------------------------------------------------------------------------

when GPU != "NONE":
  proc gpuMalloc*(size: int): GpuRef =
    ## Allocate device memory. Returns a GpuRef that auto-frees on destruction.
    result = GpuRef(size: size)
    when isHipBackend:
      handleGpuError hipMalloc(addr result.p, size.csize_t)
    elif GPU == "CUDA":
      handleGpuError cudaMalloc(addr result.p, size.csize_t)

  proc gpuFree*(p: pointer) =
    when isHipBackend:
      handleGpuError hipFree(p)
    elif GPU == "CUDA":
      handleGpuError cudaFree(p)

  # --- Memcpy variants (all 4 pointer combinations) ---

  template gpuMemcpy*(
    dst: pointer; 
    src: pointer; 
    size: int;
    kind: GpuMemcpyKind
  ) =
    when isHipBackend:
      handleGpuError hipMemcpy(dst, src, size.csize_t, kind)
    elif GPU == "CUDA":
      handleGpuError cudaMemcpy(dst, src, size.csize_t, kind)

  template gpuMemcpy*(
    dst: pointer; 
    src: GpuRef; 
    size: int;
    kind: GpuMemcpyKind
  ) =
    gpuMemcpy(dst, src.p, size, kind)

  template gpuMemcpy*(
    dst: GpuRef; 
    src: pointer; 
    size: int;
    kind: GpuMemcpyKind
  ) =
    gpuMemcpy(dst.p, src, size, kind)

  template gpuMemcpy*(
    dst: GpuRef; 
    src: GpuRef; 
    size: int;
    kind: GpuMemcpyKind
  ) =
    gpuMemcpy(dst.p, src.p, size, kind)

  template gpuMemset*(devPtr: pointer; value: int; count: int) =
    when isHipBackend:
      handleGpuError hipMemset(devPtr, value.cint, count.csize_t)
    elif GPU == "CUDA":
      handleGpuError cudaMemset(devPtr, value.cint, count.csize_t)

  template gpuMemcpyAsync*(
    dst: pointer; 
    src: pointer; 
    size: int;
    kind: GpuMemcpyKind; 
    stream: GpuStream
  ) =
    when isHipBackend:
      handleGpuError hipMemcpyAsync(dst, src, size.csize_t, kind, stream)
    elif GPU == "CUDA":
      handleGpuError cudaMemcpyAsync(dst, src, size.csize_t, kind, stream)

  # Page-locked host memory
  proc gpuHostAlloc*(size: int): pointer =
    when isHipBackend:
      handleGpuError hipHostAlloc(addr result, size.csize_t, 0.cuint)
    elif GPU == "CUDA":
      handleGpuError cudaHostAlloc(addr result, size.csize_t, 0.cuint)

  proc gpuHostFree*(p: pointer) =
    when isHipBackend:
      handleGpuError hipHostFree(p)
    elif GPU == "CUDA":
      handleGpuError cudaFreeHost(p)
else:
  proc gpuMalloc*(size: int): GpuRef = GpuRef(p: nil, size: size)

  proc gpuFree*(p: pointer) = discard
  
  template gpuMemcpy*(
    dst: pointer; 
    src: pointer; 
    size: int;
    kind: GpuMemcpyKind
  ) = discard
  
  template gpuMemset*(devPtr: pointer; value: int; count: int) = discard
  
  template gpuMemcpyAsync*(
    dst: pointer; 
    src: pointer; 
    size: int;
    kind: GpuMemcpyKind; 
    stream: GpuStream
  ) = discard
  
  proc gpuHostAlloc*(size: int): pointer = nil
  
  proc gpuHostFree*(p: pointer) = discard

# ---------------------------------------------------------------------------
# Synchronization
# ---------------------------------------------------------------------------

when GPU != "NONE":
  proc gpuSynchronize*() =
    when isHipBackend:
      handleGpuError hipDeviceSynchronize()
    elif GPU == "CUDA":
      handleGpuError cudaDeviceSynchronize()

  template gpuSyncthreads*() =
    when isHipBackend:
      hipSyncthreads()
    elif GPU == "CUDA":
      cudaSyncthreads()
else:
  proc gpuSynchronize*() = discard
  template gpuSyncthreads*() = discard

# ---------------------------------------------------------------------------
# Stream management
# ---------------------------------------------------------------------------

when GPU != "NONE":
  proc gpuStreamCreate*(): GpuStream =
    when isHipBackend:
      handleGpuError hipStreamCreate(addr result)
    elif GPU == "CUDA":
      handleGpuError cudaStreamCreate(addr result)

  proc gpuStreamDestroy*(stream: GpuStream) =
    when isHipBackend:
      handleGpuError hipStreamDestroy(stream)
    elif GPU == "CUDA":
      handleGpuError cudaStreamDestroy(stream)

  proc gpuStreamSynchronize*(stream: GpuStream) =
    when isHipBackend:
      handleGpuError hipStreamSynchronize(stream)
    elif GPU == "CUDA":
      handleGpuError cudaStreamSynchronize(stream)
else:
  proc gpuStreamCreate*(): GpuStream = nil
  proc gpuStreamDestroy*(stream: GpuStream) = discard
  proc gpuStreamSynchronize*(stream: GpuStream) = discard

# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

when GPU != "NONE":
  proc gpuEventCreate*(): GpuEvent =
    when isHipBackend:
      handleGpuError hipEventCreate(addr result)
    elif GPU == "CUDA":
      handleGpuError cudaEventCreate(addr result)

  proc gpuEventDestroy*(event: GpuEvent) =
    when isHipBackend:
      handleGpuError hipEventDestroy(event)
    elif GPU == "CUDA":
      handleGpuError cudaEventDestroy(event)

  proc gpuEventRecord*(event: GpuEvent; stream: GpuStream = nil) =
    when isHipBackend:
      handleGpuError hipEventRecord(event, stream)
    elif GPU == "CUDA":
      handleGpuError cudaEventRecord(event, stream)

  proc gpuEventSynchronize*(event: GpuEvent) =
    when isHipBackend:
      handleGpuError hipEventSynchronize(event)
    elif GPU == "CUDA":
      handleGpuError cudaEventSynchronize(event)

  proc gpuEventElapsedTime*(start: GpuEvent; stop: GpuEvent): float32 =
    var ms: cfloat
    when isHipBackend:
      handleGpuError hipEventElapsedTime(addr ms, start, stop)
    elif GPU == "CUDA":
      handleGpuError cudaEventElapsedTime(addr ms, start, stop)
    ms.float32
else:
  proc gpuEventCreate*(): GpuEvent = nil
  proc gpuEventDestroy*(event: GpuEvent) = discard
  proc gpuEventRecord*(event: GpuEvent; stream: GpuStream = nil) = discard
  proc gpuEventSynchronize*(event: GpuEvent) = discard
  proc gpuEventElapsedTime*(start: GpuEvent; stop: GpuEvent): float32 = 0.0

# ---------------------------------------------------------------------------
# Kernel launch
# ---------------------------------------------------------------------------

when GPU == "HIP" or GPU == "CUDA":
  # hipLaunchKernel / cudaLaunchKernel: takes void** args array (built via gpuArgs)
  template gpuLaunchKernel*(
    kernel: untyped;
    gridDim: Dim3 = newDim3(1, 1, 1);
    blockDim: Dim3 = newDim3(1, 1, 1);
    sharedMem: int = 0;
    stream: GpuStream = nil;
    args: untyped
  ) =
    when GPU == "HIP":
      var kernelArgs = args
      handleGpuError hipLaunchKernel(
        cast[pointer](kernel), gridDim, blockDim,
        cast[ptr pointer](addr kernelArgs[0]),
        sharedMem.csize_t, stream)
    elif GPU == "CUDA":
      var kernelArgs = args
      handleGpuError cudaLaunchKernel(
        cast[pointer](kernel), gridDim, blockDim,
        cast[ptr pointer](addr kernelArgs[0]),
        sharedMem.csize_t, stream)

  macro gpuArgs*(args: varargs[untyped]): untyped =
    ## Build an array of pointers to kernel arguments.
    var seqNode = newNimNode(nnkBracket)
    for arg in args:
      seqNode.add(quote do: cast[ptr pointer](addr `arg`))
    result = seqNode

elif GPU == "HIP_CPU":
  # HIP-CPU only provides hipLaunchKernelGGL (a macro), not hipLaunchKernel.
  # Use hipLaunchKernelGGL directly from the raw module (re-exported above)
  # with actual kernel arguments (not void** array).
  #
  #   hipLaunchKernelGGL(kernel, gridDim, blockDim, sharedMem, stream, a, b, c)
  #
  # gpuLaunchKernel / gpuArgs are not available on HIP_CPU.
  discard

else:
  template gpuLaunchKernel*(
    kernel: untyped;
    gridDim: Dim3 = newDim3(1, 1, 1);
    blockDim: Dim3 = newDim3(1, 1, 1);
    sharedMem: int = 0;
    stream: GpuStream = nil;
    args: untyped
  ) = discard
  
  macro gpuArgs*(args: varargs[untyped]): untyped =
    result = newNimNode(nnkBracket)

# ---------------------------------------------------------------------------
# Kernel attribute macros (pragmas)
# ---------------------------------------------------------------------------

macro kernel*(fn: untyped): untyped =
  ## Mark a proc as a __global__ GPU kernel (callable from host, runs on device).
  when GPU != "NONE":
    let globalPragma: NimNode = quote:
      {.exportc, codegenDecl: "__global__ $# $#$#".}
    fn.addPragma(globalPragma[0])
    fn.addPragma(globalPragma[1])
    result = quote do:
      {.push stackTrace: off, checks: off.}
      `fn`
      {.pop.}
  else:
    result = fn

macro device*(fn: untyped): untyped =
  ## Mark a proc as a __device__ function (callable from device code only).
  when GPU != "NONE":
    let devicePragma: NimNode = quote:
      {.exportc, codegenDecl: "__device__ $# $#$#".}
    fn.addPragma(devicePragma[0])
    fn.addPragma(devicePragma[1])
    result = quote do:
      {.push stackTrace: off, checks: off.}
      `fn`
      {.pop.}
  else:
    result = fn

macro hostDevice*(fn: untyped): untyped =
  ## Mark a proc as both __host__ and __device__.
  when GPU != "NONE":
    let hdPragma: NimNode = quote:
      {.exportc, codegenDecl: "__device__ __host__ $# $#$#".}
    fn.addPragma(hdPragma[0])
    fn.addPragma(hdPragma[1])
    result = quote do:
      {.push stackTrace: off, checks: off.}
      `fn`
      {.pop.}
  else:
    result = fn

macro shared*(v: untyped): untyped =
  ## Declare a variable as __shared__ memory within a kernel.
  when GPU != "NONE":
    quote do:
      {.push stackTrace: off, checks: off, noinit, exportc,
             codegenDecl: "__shared__ $# $#".}
      `v`
      {.pop.}
  else:
    quote do: `v`

macro constant*(v: untyped): untyped =
  ## Declare a variable as __constant__ (read-only, cached on-chip).
  when GPU != "NONE":
    quote do:
      {.push stackTrace: off, checks: off, exportc,
             codegenDecl: "__constant__ $# $#".}
      `v`
      {.pop.}
  else:
    quote do: `v`

# ---------------------------------------------------------------------------
# Atomics (unified names)
# ---------------------------------------------------------------------------

when GPU != "NONE":
  template gpuAtomicAdd*(address: ptr int32; val: int32): int32 =
    when isHipBackend: hipAtomicAdd(address, val)
    elif GPU == "CUDA": cudaAtomicAdd(address, val)

  template gpuAtomicAdd*(address: ptr uint32; val: uint32): uint32 =
    when isHipBackend: hipAtomicAdd(address, val)
    elif GPU == "CUDA": cudaAtomicAdd(address, val)

  template gpuAtomicSub*(address: ptr int32; val: int32): int32 =
    when isHipBackend: hipAtomicSub(address, val)
    elif GPU == "CUDA": cudaAtomicSub(address, val)

  template gpuAtomicSub*(address: ptr uint32; val: uint32): uint32 =
    when isHipBackend: hipAtomicSub(address, val)
    elif GPU == "CUDA": cudaAtomicSub(address, val)

  template gpuAtomicExch*(address: ptr int32; val: int32): int32 =
    when isHipBackend: hipAtomicExch(address, val)
    elif GPU == "CUDA": cudaAtomicExch(address, val)

  template gpuAtomicExch*(address: ptr uint32; val: uint32): uint32 =
    when isHipBackend: hipAtomicExch(address, val)
    elif GPU == "CUDA": cudaAtomicExch(address, val)

  template gpuAtomicCAS*(address: ptr int32; compare: int32; val: int32): int32 =
    when isHipBackend: hipAtomicCAS(address, compare, val)
    elif GPU == "CUDA": cudaAtomicCAS(address, compare, val)

  template gpuAtomicCAS*(address: ptr uint32; compare: uint32; val: uint32): uint32 =
    when isHipBackend: hipAtomicCAS(address, compare, val)
    elif GPU == "CUDA": cudaAtomicCAS(address, compare, val)

  template gpuAtomicMin*(address: ptr int32; val: int32): int32 =
    when isHipBackend: hipAtomicMin(address, val)
    elif GPU == "CUDA": cudaAtomicMin(address, val)

  template gpuAtomicMin*(address: ptr uint32; val: uint32): uint32 =
    when isHipBackend: hipAtomicMin(address, val)
    elif GPU == "CUDA": cudaAtomicMin(address, val)

  template gpuAtomicMax*(address: ptr int32; val: int32): int32 =
    when isHipBackend: hipAtomicMax(address, val)
    elif GPU == "CUDA": cudaAtomicMax(address, val)

  template gpuAtomicMax*(address: ptr uint32; val: uint32): uint32 =
    when isHipBackend: hipAtomicMax(address, val)
    elif GPU == "CUDA": cudaAtomicMax(address, val)

# ---------------------------------------------------------------------------
# Device math (unified names)
# ---------------------------------------------------------------------------

when GPU != "NONE":
  template gpuExp*(x: cfloat): cfloat = expf(x)
  template gpuExp*(x: cdouble): cdouble = exp(x)
  template gpuLog*(x: cfloat): cfloat = logf(x)
  template gpuLog*(x: cdouble): cdouble = log(x)
  template gpuSin*(x: cfloat): cfloat = sinf(x)
  template gpuSin*(x: cdouble): cdouble = sin(x)
  template gpuCos*(x: cfloat): cfloat = cosf(x)
  template gpuCos*(x: cdouble): cdouble = cos(x)
  template gpuSqrt*(x: cfloat): cfloat = sqrtf(x)
  template gpuSqrt*(x: cdouble): cdouble = sqrt(x)
  template gpuPow*(base: cfloat; exp: cfloat): cfloat = powf(base, exp)
  template gpuPow*(base: cdouble; exp: cdouble): cdouble = pow(base, exp)

  template gpuShflDown*(val: cfloat; delta: int): cfloat = shflDown(val, delta.cint)
  template gpuShflDown*(val: cint; delta: int): cint = shflDown(val, delta.cint)
  template gpuShfl*(val: cfloat; srcLane: int): cfloat = shfl(val, srcLane.cint)
  template gpuShfl*(val: cint; srcLane: int): cint = shfl(val, srcLane.cint)

  template gpuHalfToFloat*(h: uint16): cfloat = halfToFloat(h)
  template gpuFloatToHalf*(f: cfloat): uint16 = floatToHalf(f)

  const gpuWarpSize* = WarpSize
else:
  import std/math as stdmath
  template gpuExp*(x: cfloat): cfloat = stdmath.exp(x.float).cfloat
  template gpuExp*(x: cdouble): cdouble = stdmath.exp(x.float).cdouble
  template gpuLog*(x: cfloat): cfloat = stdmath.ln(x.float).cfloat
  template gpuLog*(x: cdouble): cdouble = stdmath.ln(x.float).cdouble
  template gpuSin*(x: cfloat): cfloat = stdmath.sin(x.float).cfloat
  template gpuSin*(x: cdouble): cdouble = stdmath.sin(x.float).cdouble
  template gpuCos*(x: cfloat): cfloat = stdmath.cos(x.float).cfloat
  template gpuCos*(x: cdouble): cdouble = stdmath.cos(x.float).cdouble
  template gpuSqrt*(x: cfloat): cfloat = stdmath.sqrt(x.float).cfloat
  template gpuSqrt*(x: cdouble): cdouble = stdmath.sqrt(x.float).cdouble
  template gpuPow*(base: cfloat; exp: cfloat): cfloat = stdmath.pow(base.float, exp.float).cfloat
  template gpuPow*(base: cdouble; exp: cdouble): cdouble = stdmath.pow(base.float, exp.float).cdouble
  const gpuWarpSize* = 1

# ---------------------------------------------------------------------------
# Device info (for locale integration)
# ---------------------------------------------------------------------------

type
  GpuDeviceInfo* = object
    id*: int
    name*: string
    totalMem*: int
    multiProcessors*: int
    warpSize*: int
    maxThreadsPerBlock*: int
    computeMajor*: int
    computeMinor*: int

proc `$`*(d: GpuDeviceInfo): string =
  &"GPU{d.id} ({d.name}, {d.totalMem div (1024*1024)} MB, " &
  &"SM {d.computeMajor}.{d.computeMinor}, {d.multiProcessors} MPs)"

proc gpuDevices*(): seq[GpuDeviceInfo] =
  ## Enumerate all GPU devices visible to this process.
  let n = gpuDeviceCount()
  result = newSeq[GpuDeviceInfo](n)
  for i in 0 ..< n:
    let prop = gpuGetDeviceProperties(i)
    result[i] = GpuDeviceInfo(
      id: i,
      name: gpuDeviceName(prop),
      totalMem: prop.totalGlobalMem.int,
      multiProcessors: prop.multiProcessorCount.int,
      warpSize: prop.warpSize.int,
      maxThreadsPerBlock: prop.maxThreadsPerBlock.int,
      computeMajor: prop.major.int,
      computeMinor: prop.minor.int,
    )
