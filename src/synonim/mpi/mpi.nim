#[
  Synonim: https://github.com/ctpeterson/Synonim
  Source file: src/synonim/mpi/mpi.nim

  Idiomatic Nim MPI wrapper. Provides type-safe communicators, automatic
  datatype mapping, error-to-exception conversion, and convenience APIs
  that will underpin the Chapel-like locale abstraction.

  The idiomatic API design (type-safe Comm/Op wrappers, mpiType mapping,
  error-to-exception conversion, and the mpiApp macro) is inspired by NimMPI.
  The abstract communicator interface pattern is informed by QEX.

  Acknowledgments:
    - NimMPI by Michalina Kotwica (Udiknedormin)
      https://github.com/Udiknedormin/NimMPI — MIT License
      Copyright (c) 2016 M. Kotwica
    - QEX (Quantum EXpressions) by James Osborn et al.
      https://github.com/jcosborn/qex — MIT License
      Copyright (c) 2015 James Osborn

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

import std/[os] 
import std/[macros]

import raw
export raw

# ---------------------------------------------------------------------------
# MPI error handling — convert C return codes to Nim exceptions
# ---------------------------------------------------------------------------

type MpiError* = object of CatchableError
  code*: cint

proc checkMpi*(err: cint) =
  if err != MPI_SUCCESS:
    var buf: array[1024, char]
    var rlen: cint
    discard MPI_Error_string(err, cast[cstring](addr buf[0]), addr rlen)
    var msg = newString(rlen)
    copyMem(addr msg[0], addr buf[0], rlen)
    var e = newException(MpiError, msg)
    e.code = err
    raise e

# ---------------------------------------------------------------------------
# Nim-idiomatic type wrappers
# ---------------------------------------------------------------------------

type
  Comm* = object
    raw*: MPI_Comm

  Op* = object
    raw*: MPI_Op

  Request* = object
    raw*: MPI_Request

  Status* = object
    raw*: MPI_Status

# Predefined communicators
let
  commWorld* = Comm(raw: MPI_COMM_WORLD)
  commSelf*  = Comm(raw: MPI_COMM_SELF)

# Predefined reduction operations
let
  opMax*     = Op(raw: MPI_MAX)
  opMin*     = Op(raw: MPI_MIN)
  opSum*     = Op(raw: MPI_SUM)
  opProd*    = Op(raw: MPI_PROD)
  opLand*    = Op(raw: MPI_LAND)
  opBand*    = Op(raw: MPI_BAND)
  opLor*     = Op(raw: MPI_LOR)
  opBor*     = Op(raw: MPI_BOR)
  opLxor*    = Op(raw: MPI_LXOR)
  opBxor*    = Op(raw: MPI_BXOR)
  opMaxloc*  = Op(raw: MPI_MAXLOC)
  opMinloc*  = Op(raw: MPI_MINLOC)
  opReplace* = Op(raw: MPI_REPLACE)

# ---------------------------------------------------------------------------
# Automatic MPI_Datatype mapping from Nim types
# ---------------------------------------------------------------------------

proc mpiType*(T: typedesc): MPI_Datatype =
  when T is int8:    MPI_INT8_T
  elif T is int16:   MPI_INT16_T
  elif T is int32:   MPI_INT32_T
  elif T is int64:   MPI_INT64_T
  elif T is uint8:   MPI_UINT8_T
  elif T is uint16:  MPI_UINT16_T
  elif T is uint32:  MPI_UINT32_T
  elif T is uint64:  MPI_UINT64_T
  elif T is float32: MPI_FLOAT
  elif T is float64: MPI_DOUBLE
  elif T is cint:    MPI_INT
  elif T is cchar:   MPI_CHAR
  elif T is char:    MPI_CHAR
  elif T is byte:    MPI_BYTE
  elif T is int:
    when sizeof(int) == 4: MPI_INT32_T
    else:                   MPI_INT64_T
  elif T is uint:
    when sizeof(uint) == 4: MPI_UINT32_T
    else:                    MPI_UINT64_T
  elif T is float:
    when sizeof(float) == 4: MPI_FLOAT
    else:                     MPI_DOUBLE
  else:
    {.error: "No MPI_Datatype mapping for " & $T.}

# ---------------------------------------------------------------------------
# Environment management
# ---------------------------------------------------------------------------

proc init*() =
  checkMpi MPI_Init(nil, nil)

proc initThread*(required: cint): cint =
  checkMpi MPI_Init_thread(nil, nil, required, addr result)

proc finalize*() =
  checkMpi MPI_Finalize()

proc initialized*(): bool =
  var flag: cint
  checkMpi MPI_Initialized(addr flag)
  flag != 0

proc finalized*(): bool =
  var flag: cint
  checkMpi MPI_Finalized(addr flag)
  flag != 0

proc wtime*(): float64 =
  MPI_Wtime()

proc wtick*(): float64 =
  MPI_Wtick()

proc processorName*(): string =
  var buf: array[256, char]
  var rlen: cint
  checkMpi MPI_Get_processor_name(cast[cstring](addr buf[0]), addr rlen)
  result = newString(rlen)
  copyMem(addr result[0], addr buf[0], rlen)

# ---------------------------------------------------------------------------
# Communicator operations
# ---------------------------------------------------------------------------

proc size*(c: Comm): int =
  var s: cint
  checkMpi MPI_Comm_size(c.raw, addr s)
  s.int

proc rank*(c: Comm): int =
  var r: cint
  checkMpi MPI_Comm_rank(c.raw, addr r)
  r.int

proc dup*(c: Comm): Comm =
  checkMpi MPI_Comm_dup(c.raw, addr result.raw)

proc split*(c: Comm, color, key: int): Comm =
  checkMpi MPI_Comm_split(c.raw, color.cint, key.cint, addr result.raw)

proc free*(c: var Comm) =
  checkMpi MPI_Comm_free(addr c.raw)

proc abort*(c: Comm, errorcode: int = 1) =
  discard MPI_Abort(c.raw, errorcode.cint)

proc isMaster*(c: Comm): bool =
  c.rank == 0

# ---------------------------------------------------------------------------
# Barrier
# ---------------------------------------------------------------------------

proc barrier*(c: Comm = commWorld) =
  checkMpi MPI_Barrier(c.raw)

# ---------------------------------------------------------------------------
# Point-to-point: send / recv
# ---------------------------------------------------------------------------

proc send*[T](
  buf: openArray[T], 
  dest: int, 
  tag: int = 0,
  comm: Comm = commWorld
) = checkMpi MPI_Send(
    unsafeAddr buf[0], 
    buf.len.cint, 
    mpiType(T),
    dest.cint, 
    tag.cint, 
    comm.raw
  )

proc send*[T](
  val: var T, 
  dest: int, 
  tag: int = 0,
  comm: Comm = commWorld
) = checkMpi MPI_Send(
    addr val, 
    1.cint, 
    mpiType(T),
    dest.cint, 
    tag.cint, 
    comm.raw
  )

proc recv*[T](
  buf: var openArray[T], 
  source: int, 
  tag: int = 0,
  comm: Comm = commWorld
): Status = checkMpi MPI_Recv(
    addr buf[0], 
    buf.len.cint, 
    mpiType(T),
    source.cint, 
    tag.cint, 
    comm.raw, 
    addr result.raw
  )

proc recv*[T](
  val: var T, 
  source: int, 
  tag: int = 0,
  comm: Comm = commWorld
): Status = checkMpi MPI_Recv(
    addr val, 
    1.cint, 
    mpiType(T),
    source.cint, 
    tag.cint, 
    comm.raw, 
    addr result.raw
  )

# ---------------------------------------------------------------------------
# Point-to-point: non-blocking send / recv
# ---------------------------------------------------------------------------

proc isend*[T](
  buf: openArray[T], dest: int, tag: int = 0,
  comm: Comm = commWorld
): Request = checkMpi MPI_Isend(
    unsafeAddr buf[0], 
    buf.len.cint, 
    mpiType(T),
    dest.cint, 
    tag.cint, 
    comm.raw, 
    addr result.raw
  )

proc irecv*[T](
  buf: var openArray[T], 
  source: int, 
  tag: int = 0,
  comm: Comm = commWorld
): Request = checkMpi MPI_Irecv(
    addr buf[0], 
    buf.len.cint, 
    mpiType(T),
    source.cint, 
    tag.cint, 
    comm.raw, 
    addr result.raw
  )

proc wait*(req: var Request): Status =
  checkMpi MPI_Wait(addr req.raw, addr result.raw)

proc waitAll*(reqs: var openArray[Request]) =
  checkMpi MPI_Waitall(
    reqs.len.cint,
    cast[ptr MPI_Request](addr reqs[0]),
    MPI_STATUSES_IGNORE
  )

proc test*(req: var Request): tuple[completed: bool, status: Status] =
  var flag: cint
  checkMpi MPI_Test(addr req.raw, addr flag, addr result.status.raw)
  result.completed = flag != 0

# ---------------------------------------------------------------------------
# Broadcast
# ---------------------------------------------------------------------------

proc bcast*[T](
  buf: var openArray[T], 
  root: int = 0,
  comm: Comm = commWorld
) = checkMpi MPI_Bcast(
    addr buf[0], 
    buf.len.cint, 
    mpiType(T),
    root.cint, 
    comm.raw
  )

proc bcast*[T](val: var T, root: int = 0, comm: Comm = commWorld) =
  checkMpi MPI_Bcast(addr val, 1.cint, mpiType(T), root.cint, comm.raw)

# ---------------------------------------------------------------------------
# Reduction operations
# ---------------------------------------------------------------------------

proc reduce*[T](
  sendbuf: openArray[T], 
  recvbuf: var openArray[T],
  op: Op, 
  root: int = 0, 
  comm: Comm = commWorld
) =
  assert sendbuf.len == recvbuf.len
  checkMpi MPI_Reduce(
    unsafeAddr sendbuf[0], 
    addr recvbuf[0],
    sendbuf.len.cint, 
    mpiType(T), 
    op.raw,
    root.cint, 
    comm.raw
  )

proc allReduce*[T](
  sendbuf: openArray[T], 
  recvbuf: var openArray[T],
  op: Op, 
  comm: Comm = commWorld
) =
  assert sendbuf.len == recvbuf.len
  checkMpi MPI_Allreduce(
    unsafeAddr sendbuf[0], 
    addr recvbuf[0],
    sendbuf.len.cint, 
    mpiType(T), 
    op.raw, 
    comm.raw
  )

proc allReduce*[T](val: var T, op: Op, comm: Comm = commWorld) =
  var tmp = val
  checkMpi MPI_Allreduce(
    addr tmp, 
    addr val, 
    1.cint, 
    mpiType(T),
    op.raw, 
    comm.raw
  )

# Convenience reduction templates
template sum*[T](val: var T, comm: Comm = commWorld) =
  allreduce(val, opSum, comm)
template max*[T](val: var T, comm: Comm = commWorld) =
  allreduce(val, opMax, comm)
template min*[T](val: var T, comm: Comm = commWorld) =
  allreduce(val, opMin, comm)

# ---------------------------------------------------------------------------
# Scatter / Gather
# ---------------------------------------------------------------------------

proc scatter*[T](
  sendbuf: openArray[T], 
  recvbuf: var openArray[T],
  root: int = 0, 
  comm: Comm = commWorld
) =
  checkMpi MPI_Scatter(
    unsafeAddr sendbuf[0], 
    recvbuf.len.cint, 
    mpiType(T),
    addr recvbuf[0], 
    recvbuf.len.cint,
    mpiType(T),
    root.cint, 
    comm.raw
  )

proc gather*[T](
  sendbuf: openArray[T], 
  recvbuf: var openArray[T],
  root: int = 0, 
  comm: Comm = commWorld
) =
  checkMpi MPI_Gather(
    unsafeAddr sendbuf[0], 
    sendbuf.len.cint, 
    mpiType(T),
    addr recvbuf[0], 
    sendbuf.len.cint, 
    mpiType(T),
    root.cint, 
    comm.raw
  )

proc allGather*[T](
  sendbuf: openArray[T], 
  recvbuf: var openArray[T],
  comm: Comm = commWorld
) =
  checkMpi MPI_Allgather(
    unsafeAddr sendbuf[0], 
    sendbuf.len.cint, 
    mpiType(T),
    addr recvbuf[0], 
    sendbuf.len.cint, 
    mpiType(T),
    comm.raw
  )

proc allToAll*[T](
  sendbuf: openArray[T], 
  recvbuf: var openArray[T],
  comm: Comm = commWorld
) =
  let n = sendbuf.len div comm.size
  checkMpi MPI_Alltoall(
    unsafeAddr sendbuf[0], 
    n.cint, 
    mpiType(T),
    addr recvbuf[0], 
    n.cint, 
    mpiType(T), 
    comm.raw
  )

# ---------------------------------------------------------------------------
# Sendrecv
# ---------------------------------------------------------------------------

proc sendRecv*[T](
  sendbuf: openArray[T], 
  dest: int, 
  sendtag: int,
  recvbuf: var openArray[T], 
  source: int, 
  recvtag: int,
  comm: Comm = commWorld
): Status =
  checkMpi MPI_Sendrecv(
    unsafeAddr sendbuf[0], 
    sendbuf.len.cint, 
    mpiType(T),
    dest.cint, 
    sendtag.cint,
    addr recvbuf[0], 
    recvbuf.len.cint, 
    mpiType(T),
    source.cint, 
    recvtag.cint,
    comm.raw, 
    addr result.raw
  )

# ---------------------------------------------------------------------------
# Cartesian topology helpers (important for Chapel-like domain distributions)
# ---------------------------------------------------------------------------

proc cartCreate*(
  comm: Comm, 
  dims: openArray[cint], 
  periods: openArray[cint],
  reorder: bool = true
): Comm =
  assert dims.len == periods.len
  checkMpi MPI_Cart_create(
    comm.raw, 
    dims.len.cint,
    unsafeAddr dims[0], 
    unsafeAddr periods[0],
    if reorder: 1.cint else: 0.cint,
    addr result.raw
  )

proc cartCoords*(comm: Comm, rank: int, ndims: int): seq[cint] =
  result = newSeq[cint](ndims)
  checkMpi MPI_Cart_coords(comm.raw, rank.cint, ndims.cint, addr result[0])

proc cartRank*(comm: Comm, coords: openArray[cint]): int =
  var r: cint
  checkMpi MPI_Cart_rank(comm.raw, unsafeAddr coords[0], addr r)
  r.int

proc cartShift*(comm: Comm, direction, disp: int): tuple[src, dest: int] =
  var s, d: cint
  checkMpi MPI_Cart_shift(comm.raw, direction.cint, disp.cint, addr s, addr d)
  (s.int, d.int)

# ---------------------------------------------------------------------------
# mpiApp pragma — inspired by NimMPI, ensures init/finalize bracketing
# ---------------------------------------------------------------------------

var mpiMainDeclared {.compileTime.}: bool = false

macro mpi*(routine: untyped): untyped =
  if mpiMainDeclared:
    error "mpiApp can only annotate one routine"
  mpiMainDeclared = true
  routine.expectKind RoutineNodes
  result = routine
  let body = routine.body
  routine.body = quote do:
    init()
    defer: finalize()
    `body`