import json
from algorithm_util import algorithm_list, spaces
from mpi4py import MPI


def load_algorithms(filename="algorithms"):
    f = open(filename, "r")
    algorithm_dict = json.load(f)
    f.close()
    return algorithm_dict


def save_algorithms(algorithm_dict={}, filename="algorithms.txt"):
    f = open(filename, "w")
    json.dump(algorithm_dict, f)
    f.close()
    return algorithm_dict


def get_algorithm_conf_from_ID(algID, algorithm_dict):
    return algorithm_dict["algID"]


def load_algorithm_with_hyperparameters(algorithm, config):
    algorithm_object = algorithm(**config)
    return algorithm_object


def create_configuration_space(name, seed=1295):
    cs = spaces[name].get_hyperparameter_search_space()
    cs.seed(seed)
    return cs


def create_algorithm_configurations(seed=1295, num_samples=64):
    algorithm_dict = {}
    num_algorithms = 0
    for algorithm, name in algorithm_list:
        cs = create_configuration_space(name, seed)
        configs = cs.sample_configuration(num_samples)
        for i in range(num_samples):
            algorithm_dict[num_samples * num_algorithms + i] = (algorithm, configs[i])
        num_algorithms = +1
    return algorithm_dict


def select_algorithms(algorithm_dict={}, num_processes=64, num_iter=0):
    selected_algorithms = {}
    for i in range(64):
        helper = num_iter * num_processes + i
        selected_algorithms[i] = algorithm_dict[helper]
    return selected_algorithms


def distribute_algorithms(
    selected_algorithms={}, num_processes=64, comm=MPI.COMM_WORLD
):
    for i in range(1, num_processes):
        req = comm.isend(selected_algorithms[i], dest=i, tag=200)
        req.wait()
    return 0


def receive_algorithm(comm=MPI.COMM_WORLD):
    req = comm.irecv(source=0, tag=200)
    data = req.wait()
    algorithm = data[0]
    config = data[1]
    return algorithm, config
