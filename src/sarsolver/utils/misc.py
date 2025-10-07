from os import sched_getaffinity


def default_num_threads() -> int:
    """
    Detects the number of cores which are available and recommends using the total number -1 or 1, whichever is higher.
    Not necessarily recommended for use on shared compute systems (e.g. HPC clusters), where it is better to specify
    the number of threads to use based upon the resources requested.
    Returns
    -------
    max_threads : int
        The number of cores on the machine minus one, or one, whichever is higher.
    """
    max_threads = len(sched_getaffinity(0))
    if max_threads > 1:
        return max_threads - 1
    else:
        return 1
