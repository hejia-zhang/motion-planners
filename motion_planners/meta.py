import time

from .lattice import lattice
from .lazy_prm import lazy_prm
from .prm import prm
from .rrt import rrt
from .rrt_connect import rrt_connect, birrt
from .rrt_star import rrt_star
from .smoothing import smooth_path, ma2_smooth_path
from .utils import RRT_RESTARTS, RRT_SMOOTHING, INF, irange, elapsed_time, compute_path_cost, compute_ma2_path_cost, \
    default_selector


def direct_path(start, goal, extend_fn, collision_fn):
    """
    :param start: Start configuration - conf
    :param goal: End configuration - conf
    :param extend_fn: Extension function - extend_fn(q1, q2)->[q', ..., q"]
    :param collision_fn: Collision function - collision_fn(q)->bool
    :return: Path [q', ..., q"] or None if unable to find a solution
    """
    # TODO: version which checks whether the segment is valid
    if collision_fn(start) or collision_fn(goal):
        # TODO: return False
        return None
    path = list(extend_fn(start, goal))
    path = [start] + path
    if any(collision_fn(q) for q in default_selector(path)):
        return None
    return path
    # path = [start]
    # for q in extend_fn(start, goal):
    #     if collision_fn(q):
    #         return None
    #     path.append(q)
    # return path


def check_direct(start, goal, extend_fn, collision_fn):
    if any(collision_fn(q) for q in [start, goal]):
        return False
    return direct_path(start, goal, extend_fn, collision_fn)


#################################################################

def random_restarts(solve_fn, start, goal, distance_fn, sample_fn, extend_fn, collision_fn,
                    restarts=RRT_RESTARTS, smooth=RRT_SMOOTHING,
                    success_cost=0., max_time=INF, max_solutions=1, **kwargs):
    """
    :param start: Start configuration - conf
    :param goal: End configuration - conf
    :param distance_fn: Distance function - distance_fn(q1, q2)->float
    :param sample_fn: Sample function - sample_fn()->conf
    :param extend_fn: Extension function - extend_fn(q1, q2)->[q', ..., q"]
    :param collision_fn: Collision function - collision_fn(q)->bool
    :param max_time: Maximum runtime - float
    :param kwargs: Keyword arguments
    :return: Paths [[q', ..., q"], [[q'', ..., q""]]
    """
    start_time = time.time()
    solutions = []
    path = check_direct(start, goal, extend_fn, collision_fn)
    if path is False:
        return None
    if path is not None:
        solutions.append(path)

    for attempt in irange(restarts + 1):
        if (len(solutions) >= max_solutions) or (elapsed_time(start_time) >= max_time):
            break
        attempt_time = (max_time - elapsed_time(start_time))
        path = solve_fn(start, goal, distance_fn, sample_fn, extend_fn, collision_fn,
                        max_time=attempt_time, **kwargs)
        if path is None:
            continue
        path = smooth_path(path, extend_fn, collision_fn, max_iterations=smooth,
                           max_time=max_time - elapsed_time(start_time))
        solutions.append(path)
        if compute_path_cost(path, distance_fn) < success_cost:
            break
    solutions = sorted(solutions, key=lambda path: compute_path_cost(path, distance_fn))
    print('Solutions ({}): {} | Time: {:.3f}'.format(len(solutions), [(len(path), round(compute_path_cost(
        path, distance_fn), 3)) for path in solutions], elapsed_time(start_time)))
    return solutions


def random_ma2_restarts(solve_fn, start_conf_pair, goal_conf_pair, distance_fn0, distance_fn1, sample_fn0, sample_fn1,
                        ma2_extend_fn, ma2_collision_fn, restarts=RRT_RESTARTS, smooth=RRT_SMOOTHING,
                        success_cost=0., max_time=INF, max_solutions=1, **kwargs):
    start_time = time.time()
    solutions = []
    path = check_direct(start_conf_pair, goal_conf_pair, ma2_extend_fn, ma2_collision_fn)
    if path is False:
        return None
    if path is not None:
        solutions.append(path)

    for attempt in irange(restarts + 1):
        if (len(solutions) >= max_solutions) or (elapsed_time(start_time) >= max_time):
            break
        attempt_time = (max_time - elapsed_time(start_time))
        path = solve_fn(start_conf_pair, goal_conf_pair, distance_fn0, distance_fn1, sample_fn0, sample_fn1,
                        ma2_extend_fn, ma2_collision_fn, max_time=attempt_time, **kwargs)
        if path is None:
            continue
        path = ma2_smooth_path(path, ma2_extend_fn, ma2_collision_fn, max_iterations=smooth,
                               max_time=max_time - elapsed_time(start_time))
        solutions.append(path)
        if compute_ma2_path_cost(path, distance_fn0, distance_fn1) < success_cost:
            break
    solutions = sorted(solutions, key=lambda path: compute_ma2_path_cost(path, distance_fn0, distance_fn1))
    print('Solutions ({}): {} | Time: {:.3f}'.format(len(solutions), [(len(path), round(compute_ma2_path_cost(
        path, distance_fn0, distance_fn1), 3)) for path in solutions], elapsed_time(start_time)))
    return solutions


def solve_and_smooth(solve_fn, q1, q2, distance_fn, sample_fn, extend_fn, collision_fn, **kwargs):
    return random_restarts(solve_fn, q1, q2, distance_fn, sample_fn, extend_fn, collision_fn, restarts=0, **kwargs)


#################################################################

def solve(start, goal, distance_fn, sample_fn, extend_fn, collision_fn, algorithm='birrt',
          max_time=INF, max_iterations=INF, num_samples=100, smooth=None, **kwargs):
    # TODO: allow distance_fn to be skipped
    # TODO: return lambda function
    start_time = time.time()
    path = check_direct(start, goal, extend_fn, collision_fn)
    if path is not None:
        return path
    # max_time -= elapsed_time(start_time)
    if algorithm == 'prm':
        path = prm(start, goal, distance_fn, sample_fn, extend_fn, collision_fn,
                   num_samples=num_samples)
    elif algorithm == 'lazy_prm':
        path = lazy_prm(start, goal, sample_fn, extend_fn, collision_fn,
                        num_samples=num_samples, max_time=max_time)[0]
    elif algorithm == 'rrt':
        path = rrt(start, goal, distance_fn, sample_fn, extend_fn, collision_fn,
                   max_iterations=max_iterations, max_time=max_time)
    elif algorithm == 'rrt_connect':
        path = rrt_connect(start, goal, distance_fn, sample_fn, extend_fn, collision_fn,
                           max_iterations=INF, max_time=max_time)
    elif algorithm == 'birrt':
        # TODO: checks the straight-line twice
        path = birrt(start, goal, distance_fn, sample_fn, extend_fn, collision_fn,
                     max_iterations=INF, max_time=max_time, smooth=None, **kwargs)  # restarts=2
    elif algorithm == 'rrt_star':
        path = rrt_star(start, goal, distance_fn, sample_fn, extend_fn, collision_fn, radius=1,
                        max_iterations=INF, max_time=max_time)
    elif algorithm == 'lattice':
        path = lattice(start, goal, extend_fn, collision_fn, distance_fn=distance_fn, max_time=INF)
    else:
        raise NotImplementedError(algorithm)
    return smooth_path(path, extend_fn, collision_fn, max_iterations=smooth,
                       max_time=max_time - elapsed_time(start_time))
