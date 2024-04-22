from mpi4py import MPI
import numpy as np
import sys


def generate_matrices(matrix_size):
    a = np.arange(0, matrix_size * matrix_size, dtype=np.float64).reshape(matrix_size, matrix_size)
    b = np.arange(0, matrix_size * matrix_size, dtype=np.float64).reshape(matrix_size, matrix_size)
    return a, b


def distribute_workload(matrix_size, comm_size):
    rows_per_process = matrix_size // comm_size
    remaining_rows = matrix_size % comm_size
    offset = 0
    workload_parts = []

    for _ in range(comm_size):
        part_size = rows_per_process + (1 if remaining_rows > 0 else 0)
        workload_parts.append([offset, offset + part_size])
        offset += part_size
        remaining_rows -= 1

    return workload_parts


def matrix_multiply(a, b):
    return np.dot(a, b)


def main():
    if len(sys.argv) != 2:
        print("Wrong args")
        return

    matrix_size = int(sys.argv[1])

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    if comm_rank == 0:
        a, b = generate_matrices(matrix_size)
        parts = distribute_workload(matrix_size, comm_size)
    else:
        a, b, parts = None, None, None

    a = comm.bcast(a, root=0)
    b = comm.bcast(b, root=0)
    parts = comm.bcast(parts, root=0)
    part = parts[comm_rank]

    start_time = MPI.Wtime()
    result = matrix_multiply(a, b[:, part[0]:part[1]])

    if comm_rank == 0:
        status = MPI.Status()
        total = np.zeros((matrix_size, matrix_size), dtype=np.float64)
        total[:, part[0]:part[1]] = result

        for i in range(1, comm_size):
            data = comm.recv(source=i, tag=0, status=status)
            total[:, parts[status.source][0]:parts[status.source][1]] = data

        end_time = MPI.Wtime()
        print(f"Time taken: {(end_time - start_time) * 1000:.4f} ms")
    else:
        comm.send(result, dest=0, tag=0)


if __name__ == "__main__":
    main()
