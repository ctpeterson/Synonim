#[
  Synonim: https://github.com/ctpeterson/Synonim
  Source file: src/synonim/examples/vectoradd.nim

  Vector-add test for the GPU device abstraction layer.
  Tests device enumeration, memory allocation, memcpy, and kernel launch.

  Compile with HIP-CPU backend (no GPU required):
    nim cpp -d:GPU=HIP_CPU -o:vectoradd src/synonim/examples/vectoradd.nim

  Compile with HIP backend (requires hipcc and GPU):
    nim cpp --cc:hipcc -d:GPU=HIP -o:vectoradd src/synonim/examples/vectoradd.nim

  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>
]#

import ../hip/device

const N = 1024

# ---------------------------------------------------------------------------
# Vector-add kernel
# ---------------------------------------------------------------------------

when GPU != "NONE":
  proc vectorAdd(A: ptr cfloat; B: ptr cfloat; C: ptr cfloat; n: cint) {.kernel.} =
    let i = blockIdx.x.int * blockDim.x.int + threadIdx.x.int
    if i < n.int:
      let ai = cast[ptr UncheckedArray[cfloat]](A)
      let bi = cast[ptr UncheckedArray[cfloat]](B)
      let ci = cast[ptr UncheckedArray[cfloat]](C)
      ci[i] = ai[i] + bi[i]

when GPU == "HIP_CPU":
  proc launchVectorAdd(grid, blk: Dim3; pA, pB, pC: pointer; n: cint) =
    {.emit: ["hipLaunchKernelGGL(vectorAdd, ",
             grid, ", ", blk, ", 0, nullptr, ",
             pA, ", ", pB, ", ", pC, ", ", n, ");"].}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

when isMainModule:
  echo "=== Synonim Vector-Add Test (GPU=" & GPU & ") ==="

  # Device info
  let nDevices = gpuDeviceCount()
  echo "GPU devices: ", nDevices
  let devs = gpuDevices()
  for d in devs:
    echo "  ", d

  # Host arrays
  var hA, hB, hC: array[N, cfloat]
  for i in 0 ..< N:
    hA[i] = i.cfloat
    hB[i] = (2 * i).cfloat
    hC[i] = 0.cfloat

  # Allocate device memory
  let bytes = N * sizeof(cfloat)
  let dA = gpuMalloc(bytes)
  let dB = gpuMalloc(bytes)
  let dC = gpuMalloc(bytes)

  # Copy host → device
  gpuMemcpy(dA.p, addr hA[0], bytes, gpuMemcpyHostToDevice)
  gpuMemcpy(dB.p, addr hB[0], bytes, gpuMemcpyHostToDevice)
  gpuMemcpy(dC.p, addr hC[0], bytes, gpuMemcpyDefault)  # zero out

  # Launch kernel
  let threadsPerBlock = 256'u32
  let blocksPerGrid = ((N.uint32 + threadsPerBlock - 1) div threadsPerBlock)
  let grid = newDim3(blocksPerGrid)
  let blk = newDim3(threadsPerBlock)
  var n: cint = N.cint

  when GPU == "HIP_CPU":
    launchVectorAdd(grid, blk, dA.p, dB.p, dC.p, n)
  elif GPU != "NONE":
    gpuLaunchKernel(vectorAdd, grid, blk, 0, nil,
                    gpuArgs(dA.p, dB.p, dC.p, n))

  gpuSynchronize()

  # Copy device → host
  gpuMemcpy(addr hC[0], dC.p, bytes, gpuMemcpyDeviceToHost)

  # Verify
  var errors = 0
  for i in 0 ..< N:
    let expected = (i + 2 * i).cfloat  # hA[i] + hB[i]
    if abs(hC[i] - expected) > 1e-5:
      if errors < 5:
        echo "  MISMATCH at ", i, ": got ", hC[i], " expected ", expected
      errors += 1

  if errors == 0:
    echo "PASSED: all ", N, " elements correct"
  else:
    echo "FAILED: ", errors, " mismatches out of ", N
    quit(1)
