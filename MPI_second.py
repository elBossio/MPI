from mpi4py import MPI
import numpy as np


def create_random_array(size):
    return np.arange(size, dtype=np.int64)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sizes = [10, 1000, 10000000]

for n in sizes:
    data = None
    startTime = 0.0
    endTime = 0.0

    if rank == 0:
        data = create_random_array(n)
        startTime = MPI.Wtime()

    local_array_size = n // size
    local_data = np.empty(local_array_size, dtype=np.int64)
    comm.Scatter([data, local_array_size, MPI.LONG_LONG], local_data, root=0)

    local_sum = np.sum(local_data)

    total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

    if rank == 0:
        endTime = MPI.Wtime()
        print(f"Total sum for array of size {n} is {total_sum}")
        print(f"Time taken: {(endTime - startTime) * 1000} ms")
