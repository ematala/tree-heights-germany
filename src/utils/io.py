from scipy.sparse import csr_matrix, load_npz, save_npz


def save_sparse_matrix(filename: str, array) -> None:
    # Convert the numpy array to a CSR sparse matrix and save it to disk
    save_npz(filename, csr_matrix(array))


def load_sparse_matrix(filename: str):
    # Load the sparse matrix from disk, convert back to a numpy array, and return
    return load_npz(filename).toarray()
