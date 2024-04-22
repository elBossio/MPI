from mpi4py import MPI
import numpy as np
import sys


def create_random_array(size):
    return np.arange(size, dtype=np.int64)


if len(sys.argv) != 2:
    print("Invalid args")
else:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    array_size = int(sys.argv[1])

    data = None
    startTime = 0.0
    endTime = 0.0

    if rank == 0:
        data = create_random_array(array_size)
        startTime = MPI.Wtime()

    local_array_size = array_size // size
    local_data = np.empty(local_array_size, dtype=np.int64)
    comm.Scatter([data, local_array_size, MPI.LONG_LONG], local_data, root=0)

    local_sum = np.sum(local_data)

    total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

    if rank == 0:
        endTime = MPI.Wtime()
        print(f"Array size is {array_size}")
        print(f"Time taken: {(endTime - startTime) * 1000} ms")
