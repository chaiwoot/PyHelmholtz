import numpy as np

# Interface to the complex-arithmetic MUMPS solver
def mumps_solver(A, b, nprocs):

    """
    Solve Ax = b using PyMUMPS in a separate MPI worker process.

    The notebook process writes A and b to a temporary file.
    MPI rank 0 loads the data and supplies the centralized matrix/RHS to MUMPS.
    """

    import os
    import pickle
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    import pyhelmholtz

    # Resolve paths independently of the notebook's current directory.
    package_dir = Path(pyhelmholtz.__file__).resolve().parent
    package_root = package_dir.parent
    worker_path = package_dir / "mumps_worker.py"

    if not worker_path.is_file():
        raise FileNotFoundError(f"MUMPS worker not found: {worker_path}")

    nprocs = int(nprocs)
    
    if nprocs < 1:
        raise ValueError("nprocs must be a positive integer.")

    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square; received shape {A.shape}.")
    
    if A.shape[0] != len(b):
        raise ValueError(
            f"Incompatible dimensions: A is {A.shape}, but b has length {len(b)}."
        )

    # The temporary directory is created under the notebook's current working directory.
    # All MPI ranks must be able to access this path.
    with tempfile.TemporaryDirectory(
        prefix="pyhelmholtz_mumps_",
        dir=os.getcwd()
    ) as temp_dir:
        
        input_path = Path(temp_dir) / "temp_input.pkl"
        output_path = Path(temp_dir) / "temp_output.pkl"

        with input_path.open("wb") as f:
            pickle.dump({"A": A, "b": np.asarray(b)}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)

        cmd = [
            "mpirun",
            "-n", str(nprocs),
            sys.executable,
            "-u",
            str(worker_path),
            str(input_path),
            str(output_path),
        ]

        result = subprocess.run(
            cmd,
            cwd=str(package_root),
            capture_output=True,
            text=True,
        )

        # enables detailed diagnostic printing/status
        # configured via ctx.set_icntl(4, 2) in mumps_worker.py
        # if result.stdout:
        #     print("--- WORKER STDOUT ---")
        #     print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")

        # if result.stderr:
        #     print("--- WORKER STDERR ---")
        #     print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")

        if result.returncode != 0:
            raise RuntimeError(
                "Parallel MUMPS solve failed.\n"
                f"Return code: {result.returncode}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

        if not output_path.is_file():
            raise RuntimeError(
                "MUMPS worker exited successfully but did not create "
                f"the solution file: {output_path}"
            )

        with output_path.open("rb") as f:
            x = pickle.load(f)

    return x

def nested_dissection_solver(A, b):
    
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu

    from sksparse.cholmod import analyze

    # Analyze the matrix A and compute the permutation vector p
    factor_structure = analyze(A.tocsc())
    p = factor_structure.P()

    # Apply permutation to A where P is the permutation matrix
    N = A.shape[0]
    P = sp.csr_matrix((np.ones(N), (np.arange(N), p)), shape=(N, N))
    A_perm = P @ A @ P.T
    b_perm = P @ b

    # Solve the linear system with SciPy's splu using 'NATURAL' ordering
    # because we've already reordered the matrix
    solver = splu(A_perm.tocsc(), permc_spec='NATURAL')
    x_perm = solver.solve(b_perm)

    # Reverse Permutation
    x = P.T @ x_perm

    return x

def gmres_solver(A, b, frequency, velocity):

    import scipy.sparse as sp
    from scipy.sparse.linalg import gmres, spilu, LinearOperator

    beta = 0.5
    k_flat = (2*np.pi*frequency/velocity).flatten() # wavenumber

    shift_diag = 1j * beta * (k_flat**2)

    M = A + sp.diags(shift_diag, format='csr')

    try:
        ilu = spilu(M.tocsc(), drop_tol=1e-3, fill_factor=10, 
                    permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.1)
        
    except RuntimeError as e:
        return None, f"Preconditioner failed: {e}"
    
    M_x = lambda x: ilu.solve(x)
    preconditioner = LinearOperator(A.shape, M_x)  # create a preconditioner
    x, info = gmres(A, b, M=preconditioner, rtol=1e-6, atol=1e-8, restart=50)

    if info != 0:
        print(f"GMRES failed to converge ({info})")

    return x
