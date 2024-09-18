import multiprocessing

def worker_function(arg):
    # Perform some computation and return a result
    result = arg ** 2
    return result

def wrapper_function(shared_array, index, arg):
    # Run the worker function and store the result in the shared array
    result = worker_function(arg)
    shared_array[index] = result

if __name__ == '__main__':
    num_processes = 4
    arguments = [1, 2, 3, 4]

    # Create a shared array to store the return values
    shared_array = multiprocessing.Array('i', num_processes)

    # Create a list to hold the processes
    processes = []

    # Spawn processes
    for i in range(num_processes):
        process = multiprocessing.Process(target=wrapper_function, args=(shared_array, i, arguments[i]))
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()

    # Retrieve the return values from the shared array
    results = [shared_array[i] for i in range(num_processes)]
    print("Return values:", results)