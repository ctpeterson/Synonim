#[
  Synonim: https://github.com/ctpeterson/Synonim
  Source file: src/synonim/mpi/raw.nim

  Raw MPI C bindings. Portable across MPI implementations (MPICH, Open MPI,
  etc.) by importing types and constants directly from mpi.h at compile time,
  rather than hardcoding implementation-specific integer values.

  The macro-based portable type import approach (defTypes/defConsts with
  {.importc, header: "mpi.h".}) is inspired by QEX. The overall structure
  of binding MPI functions with Nim pragmas is informed by NimMPI.

  Acknowledgments:
    - QEX (Quantum EXpressions) by James Osborn et al.
      https://github.com/jcosborn/qex — MIT License
      Copyright (c) 2015 James Osborn
    - NimMPI by Michalina Kotwica (Udiknedormin)
      https://github.com/Udiknedormin/NimMPI — MIT License
      Copyright (c) 2016 M. Kotwica

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

import macros
import std/[strutils]

# ---------------------------------------------------------------------------
# Compile-time MPI detection — automatically discover include/link flags
# by querying mpicc. Supports Intel MPI (`-show`), Open MPI (`-showme`),
# MPICH (`-compile-info`/`-link-info`), and pkg-config as a fallback.
# ---------------------------------------------------------------------------

proc parseMpiFlags(raw: string, kind: string): string {.compileTime.} =
  ## Extract -I flags (kind="I"), -L/-l/-Xlinker/-rpath/-Wl flags (kind="L")
  ## Handles multi-token flags like `-Xlinker --enable-new-dtags` and
  ## `-Xlinker -rpath -Xlinker /some/path` correctly.
  result = ""
  let tokens = raw.split({' ', '\t', '\n'})
  var i = 0
  while i < tokens.len:
    let t = tokens[i].strip()
    i.inc
    if t.len == 0: continue
    if kind == "I":
      if t.startsWith("-I"): result.add t & " "
    elif kind == "L":
      if t == "-Xlinker":
        # -Xlinker takes the next token as its argument
        if i < tokens.len:
          let arg = tokens[i].strip()
          i.inc
          result.add t & " " & arg & " "
      elif t.startsWith("-L") or t.startsWith("-l") or
           t.startsWith("-Wl,"):
        result.add t & " "
  result = result.strip()

const mpiShowRaw {.strdefine.} = staticExec("mpicc -show 2>/dev/null")
const mpiShowmeCompile {.strdefine.} = staticExec("mpicc -showme:compile 2>/dev/null")
const mpiShowmeLink {.strdefine.} = staticExec("mpicc -showme:link 2>/dev/null")
const mpiCompileInfo {.strdefine.} = staticExec("mpicc -compile-info 2>/dev/null")
const mpiLinkInfo {.strdefine.} = staticExec("mpicc -link-info 2>/dev/null")
const mpiPkgCflags {.strdefine.} = staticExec("pkg-config --cflags mpi 2>/dev/null")
const mpiPkgLibs {.strdefine.} = staticExec("pkg-config --libs mpi 2>/dev/null")

const mpiCflags* = block:
  var flags = ""
  # Open MPI: -showme:compile gives clean compile flags
  if mpiShowmeCompile.len > 0 and "error" notin mpiShowmeCompile.toLowerAscii():
    flags = mpiShowmeCompile.strip()
  # Intel MPI / MPICH: -show gives full cc invocation line
  elif mpiShowRaw.len > 0 and "-I" in mpiShowRaw:
    flags = parseMpiFlags(mpiShowRaw, "I")
  # MPICH: -compile-info
  elif mpiCompileInfo.len > 0 and "-I" in mpiCompileInfo:
    flags = parseMpiFlags(mpiCompileInfo, "I")
  # pkg-config fallback
  elif mpiPkgCflags.len > 0:
    flags = mpiPkgCflags.strip()
  flags

const mpiLdflags* = block:
  var flags = ""
  # Open MPI: -showme:link gives clean link flags
  if mpiShowmeLink.len > 0 and "error" notin mpiShowmeLink.toLowerAscii():
    flags = mpiShowmeLink.strip()
  # Intel MPI / MPICH: -show gives full cc invocation line
  elif mpiShowRaw.len > 0 and ("-L" in mpiShowRaw or "-lmpi" in mpiShowRaw):
    flags = parseMpiFlags(mpiShowRaw, "L")
  # MPICH: -link-info
  elif mpiLinkInfo.len > 0 and ("-L" in mpiLinkInfo or "-l" in mpiLinkInfo):
    flags = parseMpiFlags(mpiLinkInfo, "L")
  # pkg-config fallback
  elif mpiPkgLibs.len > 0:
    flags = mpiPkgLibs.strip()
  flags

when mpiCflags.len > 0:
  {.passC: mpiCflags.}
when mpiLdflags.len > 0:
  {.passL: mpiLdflags.}
when mpiCflags.len == 0 and mpiLdflags.len == 0:
  {.warning: "Could not auto-detect MPI flags. Ensure mpicc is in PATH or pass --passC/--passL manually.".}

# ---------------------------------------------------------------------------
# Compile-time helper macros (a la QEX) for portable MPI type imports.
# These import types and constants directly from mpi.h, so the bindings
# work regardless of the MPI implementation.
# ---------------------------------------------------------------------------

{.pragma: mpiH, importc, header: "mpi.h".}

macro defTypes(f: untyped, idents: untyped): untyped =
  result = newStmtList()
  for i in 0 ..< idents.len:
    result.add newCall(f, idents[i])

macro defConsts(f: untyped, idents: untyped): untyped =
  result = newStmtList()
  for i in 0 ..< idents.len:
    var t = idents[i][1]
    if t.kind == nnkStmtList: t = t[0]
    result.add newCall(f, idents[i][0], t)

# ---------------------------------------------------------------------------
# Opaque MPI types — imported as distinct Nim objects from mpi.h
# ---------------------------------------------------------------------------

template tdef(i: untyped) {.dirty.} =
  type i* {.mpiH.} = object

defTypes tdef:
  MPI_Aint
  MPI_Count
  MPI_Comm
  MPI_Datatype
  MPI_Errhandler
  MPI_File
  MPI_Group
  MPI_Info
  MPI_Op
  MPI_Request
  MPI_Message
  MPI_Status
  MPI_Win
  MPI_Fint

type MPI_Offset* {.mpiH.} = cint

# ---------------------------------------------------------------------------
# MPI constants — imported as vars from mpi.h (values are implementation
# specific and must not be hardcoded)
# ---------------------------------------------------------------------------

template cdef(i, t: untyped) {.dirty.} =
  var i* {.mpiH.}: t

defConsts cdef:
  # Communicator constants
  MPI_COMM_WORLD: MPI_Comm
  MPI_COMM_SELF: MPI_Comm
  MPI_COMM_NULL: MPI_Comm

  # Group constants
  MPI_GROUP_NULL: MPI_Group
  MPI_GROUP_EMPTY: MPI_Group

  # Special rank values
  MPI_ANY_SOURCE: cint
  MPI_PROC_NULL: cint
  MPI_ROOT: cint
  MPI_ANY_TAG: cint

  # Reduction operations
  MPI_MAX: MPI_Op
  MPI_MIN: MPI_Op
  MPI_SUM: MPI_Op
  MPI_PROD: MPI_Op
  MPI_LAND: MPI_Op
  MPI_BAND: MPI_Op
  MPI_LOR: MPI_Op
  MPI_BOR: MPI_Op
  MPI_LXOR: MPI_Op
  MPI_BXOR: MPI_Op
  MPI_MAXLOC: MPI_Op
  MPI_MINLOC: MPI_Op
  MPI_REPLACE: MPI_Op
  MPI_NO_OP: MPI_Op
  MPI_OP_NULL: MPI_Op

  # Datatypes
  MPI_DATATYPE_NULL: MPI_Datatype
  MPI_BYTE: MPI_Datatype
  MPI_PACKED: MPI_Datatype
  MPI_CHAR: MPI_Datatype
  MPI_SHORT: MPI_Datatype
  MPI_INT: MPI_Datatype
  MPI_LONG: MPI_Datatype
  MPI_FLOAT: MPI_Datatype
  MPI_DOUBLE: MPI_Datatype
  MPI_LONG_DOUBLE: MPI_Datatype
  MPI_UNSIGNED_CHAR: MPI_Datatype
  MPI_SIGNED_CHAR: MPI_Datatype
  MPI_UNSIGNED_SHORT: MPI_Datatype
  MPI_UNSIGNED: MPI_Datatype
  MPI_UNSIGNED_LONG: MPI_Datatype
  MPI_UNSIGNED_LONG_LONG: MPI_Datatype
  MPI_LONG_LONG_INT: MPI_Datatype
  MPI_INT8_T: MPI_Datatype
  MPI_UINT8_T: MPI_Datatype
  MPI_INT16_T: MPI_Datatype
  MPI_UINT16_T: MPI_Datatype
  MPI_INT32_T: MPI_Datatype
  MPI_UINT32_T: MPI_Datatype
  MPI_INT64_T: MPI_Datatype
  MPI_UINT64_T: MPI_Datatype
  MPI_C_BOOL: MPI_Datatype
  MPI_C_FLOAT_COMPLEX: MPI_Datatype
  MPI_C_DOUBLE_COMPLEX: MPI_Datatype
  MPI_FLOAT_INT: MPI_Datatype
  MPI_DOUBLE_INT: MPI_Datatype
  MPI_LONG_INT: MPI_Datatype
  MPI_SHORT_INT: MPI_Datatype
  MPI_2INT: MPI_Datatype

  # Request / status
  MPI_REQUEST_NULL: MPI_Request
  MPI_MESSAGE_NULL: MPI_Message
  MPI_MESSAGE_NO_PROC: MPI_Message
  MPI_STATUS_IGNORE: ptr MPI_Status
  MPI_STATUSES_IGNORE: ptr MPI_Status

  # Info
  MPI_INFO_NULL: MPI_Info

  # Window
  MPI_WIN_NULL: MPI_Win

  # Errhandler
  MPI_ERRHANDLER_NULL: MPI_Errhandler
  MPI_ERRORS_ARE_FATAL: MPI_Errhandler
  MPI_ERRORS_RETURN: MPI_Errhandler

  # Misc
  MPI_IN_PLACE: pointer
  MPI_BOTTOM: pointer
  MPI_BSEND_OVERHEAD: cint
  MPI_UNDEFINED: cint
  MPI_MAX_PROCESSOR_NAME: cint
  MPI_MAX_ERROR_STRING: cint
  MPI_MAX_OBJECT_NAME: cint
  MPI_MAX_LIBRARY_VERSION_STRING: cint

  # Error codes
  MPI_SUCCESS: cint
  MPI_ERR_BUFFER: cint
  MPI_ERR_COUNT: cint
  MPI_ERR_TYPE: cint
  MPI_ERR_TAG: cint
  MPI_ERR_COMM: cint
  MPI_ERR_RANK: cint
  MPI_ERR_REQUEST: cint
  MPI_ERR_ROOT: cint
  MPI_ERR_GROUP: cint
  MPI_ERR_OP: cint
  MPI_ERR_TOPOLOGY: cint
  MPI_ERR_DIMS: cint
  MPI_ERR_ARG: cint
  MPI_ERR_UNKNOWN: cint
  MPI_ERR_TRUNCATE: cint
  MPI_ERR_OTHER: cint
  MPI_ERR_INTERN: cint
  MPI_ERR_IN_STATUS: cint
  MPI_ERR_PENDING: cint
  MPI_ERR_LASTCODE: cint

  # Thread support levels
  MPI_THREAD_SINGLE: cint
  MPI_THREAD_FUNNELED: cint
  MPI_THREAD_SERIALIZED: cint
  MPI_THREAD_MULTIPLE: cint

  # Topology types
  MPI_CART: cint
  MPI_GRAPH: cint
  MPI_DIST_GRAPH: cint

  # Communicator comparison results
  MPI_IDENT: cint
  MPI_CONGRUENT: cint
  MPI_SIMILAR: cint
  MPI_UNEQUAL: cint

  # File constants
  MPI_FILE_NULL: MPI_File
  MPI_MODE_RDONLY: cint
  MPI_MODE_RDWR: cint
  MPI_MODE_WRONLY: cint
  MPI_MODE_CREATE: cint
  MPI_MODE_APPEND: cint

# ---------------------------------------------------------------------------
# MPI function bindings — environment management
# ---------------------------------------------------------------------------

proc MPI_Init*(argc: ptr cint, argv: ptr cstringArray): cint
  {.cdecl, mpiH.}

proc MPI_Init_thread*(argc: ptr cint, argv: ptr cstringArray,
                      required: cint, provided: ptr cint): cint
  {.cdecl, mpiH.}

proc MPI_Finalize*(): cint
  {.cdecl, mpiH.}

proc MPI_Initialized*(flag: ptr cint): cint
  {.cdecl, mpiH.}

proc MPI_Finalized*(flag: ptr cint): cint
  {.cdecl, mpiH.}

proc MPI_Abort*(comm: MPI_Comm, errorcode: cint): cint
  {.cdecl, mpiH.}

proc MPI_Get_processor_name*(name: cstring, resultlen: ptr cint): cint
  {.cdecl, mpiH.}

proc MPI_Get_version*(version: ptr cint, subversion: ptr cint): cint
  {.cdecl, mpiH.}

proc MPI_Wtime*(): cdouble
  {.cdecl, mpiH.}

proc MPI_Wtick*(): cdouble
  {.cdecl, mpiH.}

# ---------------------------------------------------------------------------
# MPI function bindings — communicator management
# ---------------------------------------------------------------------------

proc MPI_Comm_size*(comm: MPI_Comm, size: ptr cint): cint
  {.cdecl, mpiH.}

proc MPI_Comm_rank*(comm: MPI_Comm, rank: ptr cint): cint
  {.cdecl, mpiH.}

proc MPI_Comm_dup*(comm: MPI_Comm, newcomm: ptr MPI_Comm): cint
  {.cdecl, mpiH.}

proc MPI_Comm_split*(comm: MPI_Comm, color: cint, key: cint,
                     newcomm: ptr MPI_Comm): cint
  {.cdecl, mpiH.}

proc MPI_Comm_free*(comm: ptr MPI_Comm): cint
  {.cdecl, mpiH.}

proc MPI_Comm_create*(comm: MPI_Comm, group: MPI_Group,
                      newcomm: ptr MPI_Comm): cint
  {.cdecl, mpiH.}

proc MPI_Comm_compare*(comm1: MPI_Comm, comm2: MPI_Comm,
                       result: ptr cint): cint
  {.cdecl, mpiH.}

proc MPI_Comm_group*(comm: MPI_Comm, group: ptr MPI_Group): cint
  {.cdecl, mpiH.}

# ---------------------------------------------------------------------------
# MPI function bindings — point-to-point communication
# ---------------------------------------------------------------------------

proc MPI_Send*(buf: pointer, count: cint, datatype: MPI_Datatype,
               dest: cint, tag: cint, comm: MPI_Comm): cint
  {.cdecl, mpiH.}

proc MPI_Recv*(buf: pointer, count: cint, datatype: MPI_Datatype,
               source: cint, tag: cint, comm: MPI_Comm,
               status: ptr MPI_Status): cint
  {.cdecl, mpiH.}

proc MPI_Isend*(buf: pointer, count: cint, datatype: MPI_Datatype,
                dest: cint, tag: cint, comm: MPI_Comm,
                request: ptr MPI_Request): cint
  {.cdecl, mpiH.}

proc MPI_Irecv*(buf: pointer, count: cint, datatype: MPI_Datatype,
                source: cint, tag: cint, comm: MPI_Comm,
                request: ptr MPI_Request): cint
  {.cdecl, mpiH.}

proc MPI_Sendrecv*(sendbuf: pointer, sendcount: cint, sendtype: MPI_Datatype,
                   dest: cint, sendtag: cint,
                   recvbuf: pointer, recvcount: cint, recvtype: MPI_Datatype,
                   source: cint, recvtag: cint,
                   comm: MPI_Comm, status: ptr MPI_Status): cint
  {.cdecl, mpiH.}

proc MPI_Probe*(source: cint, tag: cint, comm: MPI_Comm,
                status: ptr MPI_Status): cint
  {.cdecl, mpiH.}

proc MPI_Iprobe*(source: cint, tag: cint, comm: MPI_Comm,
                 flag: ptr cint, status: ptr MPI_Status): cint
  {.cdecl, mpiH.}

proc MPI_Get_count*(status: ptr MPI_Status, datatype: MPI_Datatype,
                    count: ptr cint): cint
  {.cdecl, mpiH.}

# ---------------------------------------------------------------------------
# MPI function bindings — request management
# ---------------------------------------------------------------------------

proc MPI_Wait*(request: ptr MPI_Request, status: ptr MPI_Status): cint
  {.cdecl, mpiH.}

proc MPI_Waitall*(count: cint, requests: ptr MPI_Request,
                  statuses: ptr MPI_Status): cint
  {.cdecl, mpiH.}

proc MPI_Waitany*(count: cint, requests: ptr MPI_Request,
                  index: ptr cint, status: ptr MPI_Status): cint
  {.cdecl, mpiH.}

proc MPI_Test*(request: ptr MPI_Request, flag: ptr cint,
               status: ptr MPI_Status): cint
  {.cdecl, mpiH.}

proc MPI_Request_free*(request: ptr MPI_Request): cint
  {.cdecl, mpiH.}

# ---------------------------------------------------------------------------
# MPI function bindings — collective communication
# ---------------------------------------------------------------------------

proc MPI_Barrier*(comm: MPI_Comm): cint
  {.cdecl, mpiH.}

proc MPI_Bcast*(buffer: pointer, count: cint, datatype: MPI_Datatype,
                root: cint, comm: MPI_Comm): cint
  {.cdecl, mpiH.}

proc MPI_Reduce*(sendbuf: pointer, recvbuf: pointer, count: cint,
                 datatype: MPI_Datatype, op: MPI_Op,
                 root: cint, comm: MPI_Comm): cint
  {.cdecl, mpiH.}

proc MPI_Allreduce*(sendbuf: pointer, recvbuf: pointer, count: cint,
                    datatype: MPI_Datatype, op: MPI_Op,
                    comm: MPI_Comm): cint
  {.cdecl, mpiH.}

proc MPI_Scatter*(sendbuf: pointer, sendcount: cint, sendtype: MPI_Datatype,
                  recvbuf: pointer, recvcount: cint, recvtype: MPI_Datatype,
                  root: cint, comm: MPI_Comm): cint
  {.cdecl, mpiH.}

proc MPI_Gather*(sendbuf: pointer, sendcount: cint, sendtype: MPI_Datatype,
                 recvbuf: pointer, recvcount: cint, recvtype: MPI_Datatype,
                 root: cint, comm: MPI_Comm): cint
  {.cdecl, mpiH.}

proc MPI_Allgather*(sendbuf: pointer, sendcount: cint, sendtype: MPI_Datatype,
                    recvbuf: pointer, recvcount: cint, recvtype: MPI_Datatype,
                    comm: MPI_Comm): cint
  {.cdecl, mpiH.}

proc MPI_Gatherv*(sendbuf: pointer, sendcount: cint, sendtype: MPI_Datatype,
                  recvbuf: pointer, recvcounts: ptr cint, displs: ptr cint,
                  recvtype: MPI_Datatype, root: cint, comm: MPI_Comm): cint
  {.cdecl, mpiH.}

proc MPI_Scatterv*(sendbuf: pointer, sendcounts: ptr cint, displs: ptr cint,
                   sendtype: MPI_Datatype, recvbuf: pointer, recvcount: cint,
                   recvtype: MPI_Datatype, root: cint, comm: MPI_Comm): cint
  {.cdecl, mpiH.}

proc MPI_Alltoall*(sendbuf: pointer, sendcount: cint, sendtype: MPI_Datatype,
                   recvbuf: pointer, recvcount: cint, recvtype: MPI_Datatype,
                   comm: MPI_Comm): cint
  {.cdecl, mpiH.}

proc MPI_Reduce_scatter*(sendbuf: pointer, recvbuf: pointer,
                         recvcounts: ptr cint, datatype: MPI_Datatype,
                         op: MPI_Op, comm: MPI_Comm): cint
  {.cdecl, mpiH.}

proc MPI_Scan*(sendbuf: pointer, recvbuf: pointer, count: cint,
               datatype: MPI_Datatype, op: MPI_Op, comm: MPI_Comm): cint
  {.cdecl, mpiH.}

proc MPI_Exscan*(sendbuf: pointer, recvbuf: pointer, count: cint,
                 datatype: MPI_Datatype, op: MPI_Op, comm: MPI_Comm): cint
  {.cdecl, mpiH.}

# ---------------------------------------------------------------------------
# MPI function bindings — non-blocking collectives
# ---------------------------------------------------------------------------

proc MPI_Ibarrier*(comm: MPI_Comm, request: ptr MPI_Request): cint
  {.cdecl, mpiH.}

proc MPI_Ibcast*(buffer: pointer, count: cint, datatype: MPI_Datatype,
                 root: cint, comm: MPI_Comm, request: ptr MPI_Request): cint
  {.cdecl, mpiH.}

proc MPI_Ireduce*(sendbuf: pointer, recvbuf: pointer, count: cint,
                  datatype: MPI_Datatype, op: MPI_Op, root: cint,
                  comm: MPI_Comm, request: ptr MPI_Request): cint
  {.cdecl, mpiH.}

proc MPI_Iallreduce*(sendbuf: pointer, recvbuf: pointer, count: cint,
                     datatype: MPI_Datatype, op: MPI_Op, comm: MPI_Comm,
                     request: ptr MPI_Request): cint
  {.cdecl, mpiH.}

# ---------------------------------------------------------------------------
# MPI function bindings — user-defined operations
# ---------------------------------------------------------------------------

type MPI_User_function* = proc (invec, inoutvec: pointer,
                                len: ptr cint, datatype: ptr MPI_Datatype)
  {.cdecl.}

proc MPI_Op_create*(function: MPI_User_function, commute: cint,
                    op: ptr MPI_Op): cint
  {.cdecl, mpiH.}

proc MPI_Op_free*(op: ptr MPI_Op): cint
  {.cdecl, mpiH.}

# ---------------------------------------------------------------------------
# MPI function bindings — group management
# ---------------------------------------------------------------------------

proc MPI_Group_size*(group: MPI_Group, size: ptr cint): cint
  {.cdecl, mpiH.}

proc MPI_Group_rank*(group: MPI_Group, rank: ptr cint): cint
  {.cdecl, mpiH.}

proc MPI_Group_incl*(group: MPI_Group, n: cint, ranks: ptr cint,
                     newgroup: ptr MPI_Group): cint
  {.cdecl, mpiH.}

proc MPI_Group_excl*(group: MPI_Group, n: cint, ranks: ptr cint,
                     newgroup: ptr MPI_Group): cint
  {.cdecl, mpiH.}

proc MPI_Group_free*(group: ptr MPI_Group): cint
  {.cdecl, mpiH.}

# ---------------------------------------------------------------------------
# MPI function bindings — Cartesian topology
# ---------------------------------------------------------------------------

proc MPI_Cart_create*(comm: MPI_Comm, ndims: cint, dims: ptr cint,
                      periods: ptr cint, reorder: cint,
                      comm_cart: ptr MPI_Comm): cint
  {.cdecl, mpiH.}

proc MPI_Cart_coords*(comm: MPI_Comm, rank: cint, maxdims: cint,
                      coords: ptr cint): cint
  {.cdecl, mpiH.}

proc MPI_Cart_rank*(comm: MPI_Comm, coords: ptr cint, rank: ptr cint): cint
  {.cdecl, mpiH.}

proc MPI_Cart_shift*(comm: MPI_Comm, direction: cint, disp: cint,
                     rank_source: ptr cint, rank_dest: ptr cint): cint
  {.cdecl, mpiH.}

proc MPI_Dims_create*(nnodes: cint, ndims: cint, dims: ptr cint): cint
  {.cdecl, mpiH.}

proc MPI_Cartdim_get*(comm: MPI_Comm, ndims: ptr cint): cint
  {.cdecl, mpiH.}

# ---------------------------------------------------------------------------
# MPI function bindings — datatype management
# ---------------------------------------------------------------------------

proc MPI_Type_contiguous*(count: cint, oldtype: MPI_Datatype,
                          newtype: ptr MPI_Datatype): cint
  {.cdecl, mpiH.}

proc MPI_Type_vector*(count: cint, blocklength: cint, stride: cint,
                      oldtype: MPI_Datatype, newtype: ptr MPI_Datatype): cint
  {.cdecl, mpiH.}

proc MPI_Type_create_subarray*(ndims: cint, sizes: ptr cint, subsizes: ptr cint,
                               starts: ptr cint, order: cint,
                               oldtype: MPI_Datatype,
                               newtype: ptr MPI_Datatype): cint
  {.cdecl, mpiH.}

proc MPI_Type_commit*(datatype: ptr MPI_Datatype): cint
  {.cdecl, mpiH.}

proc MPI_Type_free*(datatype: ptr MPI_Datatype): cint
  {.cdecl, mpiH.}

proc MPI_Type_size*(datatype: MPI_Datatype, size: ptr cint): cint
  {.cdecl, mpiH.}

# ---------------------------------------------------------------------------
# MPI function bindings — error handling
# ---------------------------------------------------------------------------

proc MPI_Error_string*(errorcode: cint, str: cstring, resultlen: ptr cint): cint
  {.cdecl, mpiH.}

proc MPI_Error_class*(errorcode: cint, errorclass: ptr cint): cint
  {.cdecl, mpiH.}

# ---------------------------------------------------------------------------
# MPI function bindings — one-sided (RMA) communication
# (Needed for Chapel-style remote execution via `on` statements)
# ---------------------------------------------------------------------------

proc MPI_Win_create*(base: pointer, size: MPI_Aint, disp_unit: cint,
                     info: MPI_Info, comm: MPI_Comm,
                     win: ptr MPI_Win): cint
  {.cdecl, mpiH.}

proc MPI_Win_allocate*(size: MPI_Aint, disp_unit: cint, info: MPI_Info,
                       comm: MPI_Comm, baseptr: pointer,
                       win: ptr MPI_Win): cint
  {.cdecl, mpiH.}

proc MPI_Win_free*(win: ptr MPI_Win): cint
  {.cdecl, mpiH.}

proc MPI_Win_fence*(assert: cint, win: MPI_Win): cint
  {.cdecl, mpiH.}

proc MPI_Put*(origin_addr: pointer, origin_count: cint,
              origin_datatype: MPI_Datatype, target_rank: cint,
              target_disp: MPI_Aint, target_count: cint,
              target_datatype: MPI_Datatype, win: MPI_Win): cint
  {.cdecl, mpiH.}

proc MPI_Get*(origin_addr: pointer, origin_count: cint,
              origin_datatype: MPI_Datatype, target_rank: cint,
              target_disp: MPI_Aint, target_count: cint,
              target_datatype: MPI_Datatype, win: MPI_Win): cint
  {.cdecl, mpiH.}

proc MPI_Win_lock*(lock_type: cint, rank: cint, assert: cint,
                   win: MPI_Win): cint
  {.cdecl, mpiH.}

proc MPI_Win_unlock*(rank: cint, win: MPI_Win): cint
  {.cdecl, mpiH.}

proc MPI_Win_lock_all*(assert: cint, win: MPI_Win): cint
  {.cdecl, mpiH.}

proc MPI_Win_unlock_all*(win: MPI_Win): cint
  {.cdecl, mpiH.}

proc MPI_Win_flush*(rank: cint, win: MPI_Win): cint
  {.cdecl, mpiH.}

proc MPI_Win_flush_all*(win: MPI_Win): cint
  {.cdecl, mpiH.}

proc MPI_Accumulate*(origin_addr: pointer, origin_count: cint,
                     origin_datatype: MPI_Datatype, target_rank: cint,
                     target_disp: MPI_Aint, target_count: cint,
                     target_datatype: MPI_Datatype, op: MPI_Op,
                     win: MPI_Win): cint
  {.cdecl, mpiH.}
