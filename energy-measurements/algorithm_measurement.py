import sklearn.datasets
import sklearn.model_selection
from mpi4py import MPI
import configurations as cf
import time
import numpy as np


class experiment:
    def __init__(
        self, filename, name, algorithm, config, data_X, data_y, comm=MPI.COMM_WORLD
    ):
        self.filename = filename
        self.name = name
        self.algorithm = algorithm
        self.config = config
        (
            self.trainingset_X,
            self.testset_X,
            self.trainingset_y,
            self.testset_y,
        ) = sklearn.model_selection.train_test_split(
            data_X, data_y, test_size=0.3, train_size=0.7
        )
        self.comm = MPI.COMM_WORLD

    # not thread safe
    # not safe to execute before runtime was measured
    def log_rank_and_time(self, rank):
        f = open(self.filename, "w")
        f.write(f"{self.name};{rank};{self.starttime};{self.endtime};{self.runtime} \n")
        f.close()
        return 0

    # not thread safe
    # not safe to execute before runtime was measured
    def log_instance(self, rank, iterations):
        f = open(self.filename, "w")
        f.write(
            f"{self.name};{rank};{iterations}{self.starttime};{self.endtime};{self.runtime};{self.config} \n"
        )
        f.close()
        return 0

    def measure_runtime(self, iterations=1, rank=0):
        self.algorithmobject = cf.load_algorithm_with_hyperparameters(
            self.algorithm, self.config
        )
        self.starttime = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        for iteration in range(iterations):
            self.algorithmobject.fit(self.trainingset_X, self.trainingset_y)
        self.endtime = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        self.runtime = self.endtime - self.starttime
        return 0

    def get_runtime(self):
        return self.runtime


def iterations_from_least_common_multiple(runtimes: list[int]):
    # round to the nearest 1000 nanosecondss
    runtimes = np.round(runtimes, -3)
    lcm = np.lcm.reduce(runtimes)
    print(f"Least Common Multiplier is: {lcm}")
    iterations = lcm / runtimes
    return iterations


def single_core_full_load(rank):
    data_X, data_y = sklearn.datasets.load_iris(return_X_y=True)
    size = MPI.COMM_WORLD.Get_size()
    if rank == 0:
        algorithm_dict = cf.create_algorithm_configurations()
        selected_confs = cf.select_algorithms(algorithm_dict, size)
        cf.distribute_algorithms(selected_confs, size)
        algorithm, config = selected_confs[0]

    else:
        algorithm, config = cf.receive_algorithm()

    instance = experiment(
        f"logs/testlog_{rank}", "test", algorithm, config, data_X, data_y
    )
    instance.measure_runtime(rank=rank)
    runtimes = MPI.COMM_WORLD.gather(instance.get_runtime())
    if rank == 0:
        iterations = iterations_from_least_common_multiple(runtimes)
    MPI.COMM_WORLD.bcast(iterations)
    instance.measure_runtime(iterations=iterations[rank], rank=rank)
    instance.log_instance(rank=rank, iterations=iterations[rank])
    return 0


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    single_core_full_load(rank)

    # get_runtime(algID)

    return 0


if __name__ == "__main__":
    main()
