"""MPI worker for solving a centralized sparse system with PyMUMPS"""

import pickle
import sys
import traceback

import numpy as np
from mpi4py import MPI
from mumps import ZMumpsContext

import pyhelmholtz as ph


def run_worker():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if len(sys.argv) != 3:
        raise RuntimeError(
            "Usage: mumps_worker.py <input_path> <output_path>"
        )

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Only rank 0 reads the full CSR matrix and RHS.
    # This avoids keeping an unnecessary full copy of the input matrix on every MPI process.
    A = None
    b = None
    load_error = None

    if rank == 0:

        try:

            with open(input_path, "rb") as f:
                data = pickle.load(f)

            A = data["A"]
            b = np.asarray(data["b"])

            if A.shape[0] != A.shape[1]:
                raise ValueError(f"A must be square; got {A.shape}.")
            
            if A.shape[0] != b.size:
                raise ValueError(
                    f"Incompatible dimensions: A={A.shape}, b={b.shape}."
                )
            
            if not np.isfinite(A.data).all():
                raise ValueError("Matrix A contains NaN or Inf.")
            
            if not np.isfinite(b).all():
                raise ValueError("Vector b contains NaN or Inf.")

        except Exception:
            load_error = traceback.format_exc()

    # Ensure all ranks agree whether input loading succeeded before entering MUMPS collectively.
    load_error = comm.bcast(load_error, root=0)

    if load_error is not None:
        raise RuntimeError(f"Input loading failed on rank 0:\n{load_error}")

    # Initialize the MUMPS context on every MPI rank.
    ctx = ZMumpsContext(sym=0, par=1)

    # enables detailed diagnostic printing/status
    ctx.set_icntl(4, 2) 

    # enable out-of-core factorization
    ctx.set_icntl(22, 1)  

    x = None
    preparation_error = None

    try:

        # PyMUMPS centralized matrix input
        # Rank 0 supplies the matrix and RHS
        # All ranks participate in the subsequent MUMPS jobs.
        if rank == 0:

            try:
                x = np.array(b, dtype=np.complex128, copy=True)
                n = int(b.size)

                row, col = ph.Util.get_row_col_indices_of_csr_matrix(A)

                # MUMPS uses 1-based matrix indices.
                # This int32 conversion is appropriate only when the installed MUMPS interface uses 32-bit MUMPS_INT and the indices fit that range.
                irn = np.asarray(row, dtype=np.int32) + np.int32(1)
                jcn = np.asarray(col, dtype=np.int32) + np.int32(1)
                aval = np.asarray(A.data, dtype=np.complex128)

                if irn.size != aval.size or jcn.size != aval.size:
                    raise ValueError("CSR coordinate arrays and matrix data have different lengths.")

                ctx.set_shape(n)
                ctx.set_centralized_assembled(irn, jcn, aval)
                ctx.set_rhs(x)

            except Exception:
                preparation_error = traceback.format_exc()

        # Avoid other ranks entering a MUMPS job if rank 0 failed while preparing or supplying the centralized input.
        preparation_error = comm.bcast(preparation_error, root=0)
        if preparation_error is not None:

            raise RuntimeError(
                "MUMPS input preparation failed on rank 0:\n"
                f"{preparation_error}"
            )

        selected_algorithm = 0 # 0=AMD, 6=QAMD, 4=PORD
        ctx.set_icntl(7, selected_algorithm)

        ctx.run(job=6) # Run Analysis, Factorization, and Solve together

        # With centralized RHS input, rank 0 owns the resulting solution.
        # Save it before destroying the context.
        # No extra Bcast is needed because only rank 0 writes the output file.
        save_error = None
        if rank == 0:

            try:
                with open(output_path, "wb") as f:
                    pickle.dump(x, f, protocol=pickle.HIGHEST_PROTOCOL)

            except Exception:
                save_error = traceback.format_exc()

        save_error = comm.bcast(save_error, root=0)
        if save_error is not None:
            raise RuntimeError(f"Saving solution failed:\n{save_error}")

    finally:
        ctx.destroy()

if __name__ == "__main__":

    try:
        run_worker()

    except BaseException:
        print(
            f"[Rank {MPI.COMM_WORLD.Get_rank()}] Worker terminated "
            "with an exception:",
            flush=True,
        )
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
