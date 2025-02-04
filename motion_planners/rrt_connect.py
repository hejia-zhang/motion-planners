import time

from .primitives import extend_towards, ma2_extend_towards
from .rrt import TreeNode, configs
from .utils import irange, RRT_ITERATIONS, INF, elapsed_time


def wrap_collision_fn(collision_fn):
    # TODO: joint limits
    # import inspect
    # print(inspect.getargspec(collision_fn))
    # print(dir(collision_fn))
    def fn(q1, q2):
        try:
            return collision_fn(q1, q2)
        except TypeError:
            return collision_fn(q2)

    return fn


def rrt_connect(start, goal, distance_fn, sample_fn, extend_fn, collision_fn,
                max_iterations=RRT_ITERATIONS, max_time=INF, **kwargs):
    """
    :param start: Start configuration - conf
    :param goal: End configuration - conf
    :param distance_fn: Distance function - distance_fn(q1, q2)->float
    :param sample_fn: Sample function - sample_fn()->conf
    :param extend_fn: Extension function - extend_fn(q1, q2)->[q', ..., q"]
    :param collision_fn: Collision function - collision_fn(q)->bool
    :param max_iterations: Maximum number of iterations - int
    :param max_time: Maximum runtime - float
    :param kwargs: Keyword arguments
    :return: Path [q', ..., q"] or None if unable to find a solution
    """
    # TODO: goal sampling function connected to a None node
    start_time = time.time()
    if collision_fn(start) or collision_fn(goal):
        return None
    # TODO: support continuous collision_fn with two arguments
    # collision_fn = wrap_collision_fn(collision_fn)
    nodes1, nodes2 = [TreeNode(start)], [TreeNode(goal)]  # TODO: allow a tree to be prespecified (possibly as start)
    for iteration in irange(max_iterations):
        if elapsed_time(start_time) >= max_time:
            break
        swap = len(nodes1) > len(nodes2)
        tree1, tree2 = nodes1, nodes2
        if swap:
            tree1, tree2 = nodes2, nodes1

        target = sample_fn()
        last1, _ = extend_towards(tree1, target, distance_fn, extend_fn, collision_fn,
                                  swap, **kwargs)
        last2, success = extend_towards(tree2, last1.config, distance_fn, extend_fn, collision_fn,
                                        not swap, **kwargs)

        if success:
            path1, path2 = last1.retrace(), last2.retrace()
            if swap:
                path1, path2 = path2, path1
            # print('{} max_iterations, {} nodes'.format(iteration, len(nodes1) + len(nodes2)))
            path = configs(path1[:-1] + path2[::-1])
            # TODO: return the trees
            return path
    return None


def ma2_rrt_connect(start_conf_pair, goal_conf_pair, distance_fn0, distance_fn1, sample_fn0, sample_fn1,
                    ma2_extend_fn, ma2_collision_fn,
                    max_iterations=RRT_ITERATIONS, max_time=INF, **kwargs):
    # TODO: goal sampling function connected to a None node
    start_time = time.time()
    if ma2_collision_fn(start_conf_pair) or ma2_collision_fn(goal_conf_pair):
        return None
    # TODO: support continuous collision_fn with two arguments
    # collision_fn = wrap_collision_fn(collision_fn)
    nodes1, nodes2 = [TreeNode(start_conf_pair)], [
        TreeNode(goal_conf_pair)]  # TODO: allow a tree to be prespecified (possibly as start)
    for iteration in irange(max_iterations):
        if elapsed_time(start_time) >= max_time:
            break
        swap = len(nodes1) > len(nodes2)
        tree1, tree2 = nodes1, nodes2
        if swap:
            tree1, tree2 = nodes2, nodes1

        (target0, target1) = sample_fn0(), sample_fn1()
        last1, _ = ma2_extend_towards(tree1, (target0, target1), distance_fn0, distance_fn1, ma2_extend_fn,
                                      ma2_collision_fn, swap, **kwargs)
        last2, success = ma2_extend_towards(tree2, last1.config, distance_fn0, distance_fn1, ma2_extend_fn,
                                            ma2_collision_fn, not swap, **kwargs)

        if success:
            path1, path2 = last1.retrace(), last2.retrace()
            if swap:
                path1, path2 = path2, path1
            # print('{} max_iterations, {} nodes'.format(iteration, len(nodes1) + len(nodes2)))
            path = configs(path1[:-1] + path2[::-1])
            # TODO: return the trees
            return path
    return None


#################################################################

def birrt(start, goal, distance_fn, sample_fn, extend_fn, collision_fn, **kwargs):
    """
    :param start: Start configuration - conf
    :param goal: End configuration - conf
    :param distance_fn: Distance function - distance_fn(q1, q2)->float
    :param sample_fn: Sample function - sample_fn()->conf
    :param extend_fn: Extension function - extend_fn(q1, q2)->[q', ..., q"]
    :param collision_fn: Collision function - collision_fn(q)->bool
    :param kwargs: Keyword arguments
    :return: Path [q', ..., q"] or None if unable to find a solution
    """
    # TODO: deprecate
    from .meta import random_restarts
    solutions = random_restarts(rrt_connect, start, goal, distance_fn, sample_fn, extend_fn, collision_fn,
                                max_solutions=1, **kwargs)
    if not solutions:
        return None
    return solutions[0]


def ma2_birrt(start_conf_pair, goal_conf_pair, distance_fn0, distance_fn1, sample_fn0, sample_fn1,
              ma2_extend_fn, ma2_collision_fn, **kwargs):
    """
    :param start_conf_pair: Start configuration - (conf0, conf1)
    :param goal_conf_pair: End configuration - (conf0, conf1)
    :param distance_fn0: Distance function - distance_fn(q1, q2)->float
    :param distance_fn1: Distance function - distance_fn(q1, q2)->float
    :param sample_fn0: Sample function - sample_fn()->conf
    :param sample_fn1: Sample function - sample_fn()->conf
    :param ma2_extend_fn: Extension function - extend_fn((q0_1, q1_1), (q0_2, q1_2))->[(q0', q1'), ..., (q0", q1")]
    :param ma2_collision_fn: Collision function - collision_fn((q0, q1))->bool
    :param kwargs: Keyword arguments
    :return: Path [(q0', q1'), ..., (q0", q1")] or None if unable to find a solution
    """
    # TODO: deprecate
    from .meta import random_ma2_restarts
    solutions = random_ma2_restarts(ma2_rrt_connect, start_conf_pair, goal_conf_pair, distance_fn0, distance_fn1,
                                    sample_fn0, sample_fn1, ma2_extend_fn, ma2_collision_fn,
                                    max_solutions=1, **kwargs)
    if not solutions:
        return None
    return solutions[0]
