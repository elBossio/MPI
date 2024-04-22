from mpi4py import MPI
import numpy as np


def f(x, y):
    return np.cos(x * x) * np.sin(y)


def dfdy(x, y):
    return np.cos(x * x) * np.cos(y)


def calc(size, row):
    matrix = np.zeros(size)
    for col in range(size):
        matrix[col] = dfdy(row, col)

    return matrix


def root(matrix_size, comm_size, tag):
    start_time = MPI.Wtime()
    sent = 0
    received = 0
    matrix = [None] * matrix_size

    while True:
        for dest in range(1, comm_size):
            MPI.COMM_WORLD.send(sent, dest=dest, tag=tag)
            sent += 1
            if sent == matrix_size:
                break

        current = received
        for dest in range(1, comm_size):
            buffer = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=tag)
            matrix[current + dest - 1] = buffer
            received += 1
            if received == matrix_size:
                break

        if sent == matrix_size and received == matrix_size:
            break

    for dest in range(1, comm_size):
        MPI.COMM_WORLD.send(-1, dest=dest, tag=tag)

    end_time = MPI.Wtime()
    total_time = (end_time - start_time) * 1000
    print(f"Calculation complete in {total_time:.2f} milliseconds")


def node(matrix_size, tag):
    while True:
        row = MPI.COMM_WORLD.recv(source=0, tag=tag)
        if row == -1:
            break
        matrix = calc(matrix_size, row)
        MPI.COMM_WORLD.send(matrix, dest=0, tag=tag)


def main():
    import sys

    if len(sys.argv) != 2:
        print("Invalid args")
        return

    matrix_size = int(sys.argv[1])
    comm_rank = MPI.COMM_WORLD.Get_rank()
    comm_size = MPI.COMM_WORLD.Get_size()
    tag = 0

    if comm_rank == 0:
        root(matrix_size, comm_size, tag)
    else:
        node(matrix_size, tag)


if __name__ == "__main__":
    main()
