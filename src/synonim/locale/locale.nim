#[
  Synonim: https://github.com/ctpeterson/Synonim
  Source file: src/synonim/locale/locale.nim
  
  Chapel-like locale abstraction backed by MPI. Each MPI rank corresponds
  to one locale. Locale names are gathered from all ranks via
  MPI_Get_processor_name (i.e. the hostname).

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

import std/[cpuinfo]

import ../mpi/[mpi]
import ../hip/[device]

type
  Locale* = object
    id*: int
    name*: string
    numCores*: int
    gpus*: seq[GpuDeviceInfo]

proc `$`*(loc: Locale): string =
  result = "LOCALE" & $loc.id & " (" & loc.name & ", " & $loc.numCores & " cores"
  if loc.gpus.len > 0:
    result.add(", " & $loc.gpus.len & " GPU" & (if loc.gpus.len > 1: "s" else: ""))
  result.add(")")

proc numLocales*(comm: Comm = commWorld): int = comm.size()

proc here*(comm: Comm = commWorld): Locale =
  Locale(
    id: comm.rank(),
    name: processorName(),
    numCores: countProcessors(),
    gpus: gpuDevices()
  )

proc locales*(comm: Comm = commWorld): seq[Locale] =
  ## Gather locale info from all ranks. Each rank contributes its hostname.
  ## The result is broadcast so every rank has the full list.
  let n = comm.size()

  # Fixed-size name buffer for allgather
  const maxName = 256
  var myNameBuf: array[maxName, byte]
  let myName = processorName()
  for i in 0..<min(myName.len, maxName):
    myNameBuf[i] = myName[i].byte

  # Allgather the name buffers
  var allNames = newSeq[byte](n * maxName)
  allGather(myNameBuf.toOpenArray(0, maxName - 1), allNames, comm)

  # Allgather core counts
  var myCores = [countProcessors().int32]
  var allCores = newSeq[int32](n)
  allGather(myCores.toOpenArray(0, 0), allCores, comm)

  # Allgather GPU counts
  var myGpuCount = [gpuDeviceCount().int32]
  var allGpuCounts = newSeq[int32](n)
  allgather(myGpuCount.toOpenArray(0, 0), allGpuCounts, comm)

  # Local GPU info (only meaningful for this rank's locale)
  let localGpus = gpuDevices()

  result = newSeq[Locale](n)
  for i in 0 ..< n:
    let offset = i * maxName
    var name = ""
    for j in 0 ..< maxName:
      let c = allNames[offset + j]
      if c == 0: break
      name.add(char(c))
    # Each rank gets its own GPU info for its locale; other locales get count only
    let gpus = if i == comm.rank(): localGpus else: newSeq[GpuDeviceInfo](0)
    result[i] = Locale(id: i, name: name, numCores: allCores[i].int, gpus: gpus)

when isMainModule:
  proc main() {.mpi.} =
    let locs = locales()
    let me = here()
    if commWorld.isMaster:
      echo "Number of locales: ", locs.len
      for loc in locs: echo "  ", loc
    barrier()
    echo "here on rank ", me.id, ": ", me
    if me.gpus.len > 0:
      for gpu in me.gpus:
        echo "  ", gpu
    else:
      echo "  (no GPUs)"

  main()