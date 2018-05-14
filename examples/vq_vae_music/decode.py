import numpy as np
import itertools

def dijkstra_path(graph, start, end, visited=[], distances={}, predecessors={}):
    """Find the shortest path between start and end nodes in a graph"""
    # we've found our end node, now find the path to it, and return
    if start == end:
        path = []
        while end != None:
            path.append(end)
            end = predecessors.get(end, None)
        return distances[start], path[::-1]
    # detect if it's the first time through, set current distance to zero
    if not visited:
        distances[start] = 0

    # process neighbors as per algorithm, keep track of predecessors
    for neighbor in graph[start]:
        if neighbor not in visited:
            neighbordist = distances.get(neighbor, float("inf"))
            tentativedist = distances[start] + graph[start][neighbor]
            if tentativedist < neighbordist:
                distances[neighbor] = tentativedist
                predecessors[neighbor] = start
    # neighbors processed, now mark the current node as visited
    visited.append(start)
    # finds the closest unvisited node to the start
    unvisiteds = dict((k, distances.get(k, float("inf"))) for k in graph if k not in visited)
    closestnode = min(unvisiteds, key=unvisiteds.get)
    # now we can take the closest node and recurse, making it current
    return dijkstra_path(graph, closestnode, end, visited, distances, predecessors)


def decode_measure(measure, last_values=None):
    # decode a measure with shortest path to discover the best greedy assignment
    # of notes to voices
    measure_len = measure.shape[1]
    all_up = list(zip(*np.where(measure)))
    time_ordered = [au for i in range(measure_len) for au in all_up if au[1] == i]
    events = {}
    for to in time_ordered:
        if to[1] not in events:
            events[to[1]] = []
        events[to[1]].append(to[0])

    for k in events.keys():
        if len(events[k]) < 4:
            tt = events[k] + [0] * 4
            tt = tt[:4]
            events[k] = tt

    voices = []
    for v in range(4):
        voices.append([])
    for ts in range(measure_len):
        # currently ordered bass -> soprano
        for v in range(4):
            if ts in events:
                voices[v].append(events[ts][v])
            else:
                voices[v].append(0)
    # return voices ordered soprano, alto, tenor, bass
    voices = voices[::-1]
    return voices
    from IPython import embed; embed(); raise ValueError()
    # NONE OF THIS WORKS YET

    # graph in dict of dict structure
    cost_graph = {}

    # add initial
    aa = list(itertools.product(range(4), repeat=4))
    aa = [aai for aai in aa if len(set(aai)) == len(aai)]
    # doubles as a prior
    if last_values == None:
        last_values = events[0]
    cost_graph[-1] = {}
    for p1 in aa:
        movement_cost = 1 + sum([last_values[m] != events[0][p1[m]] for m in range(len(p1))])
        transition_cost = sum([abs(last_values[m] - events[0][p1[m]]) 
                              for m in range(len(p1))])
        cost_graph[-1][(0,) + p1] = movement_cost * transition_cost

    for i in range(measure_len - 1):
        j = i + 1
        this_e = events[i]
        next_e = events[j]

        # all pairwise costs - not allowed repeats
        # (i, (a, a, a, a)) -> (j, (a, a, a, a))
        aa = list(itertools.product(range(4), repeat=4))
        aa = [aai for aai in aa if len(set(aai)) == len(aai)]
        # keys are
        # (i, a, a, a, a) -> (j, a, a, a, a)
        for p1 in aa:
            cost_graph[(i,) + p1] = {}
            for p2 in aa:
                movement_cost = 1 + sum([this_e[p1[m]] != next_e[p2[m]] for m in range(len(p1))])
                transition_cost = sum([abs(this_e[p1[m]] - next_e[p2[m]])
                                       for m in range(len(p1))])
                cost_graph[(i,) + p1][(j,) + p2] = movement_cost * transition_cost
                if j == (measure_len - 1):
                    # closure node
                    cost_graph[(j,) + p2] = {}
                    cost_graph[(j,) + p2][measure_len] = 0
    # closure
    cost_graph[measure_len] = {0: 0}

    # do multi-step greedy? assign 1 voice, next voice, next voice, next voice?
    #print("running dijkstra")
    #print("running decode")
    r = dijkstra_path(cost_graph, -1, measure_len)
    assignments = r[1][1:-1]
    voices = []
    for v in range(4):
        voices.append([])
    for ts in range(measure_len):
        # currently ordered bass -> soprano
        for v in range(4):
            # offset by one because the key is (ts, a, a, a, a)
            voices[v].append(events[ts][assignments[ts][v + 1]])
    # return voices ordered soprano, alto, tenor, bass
    voices = voices[::-1]
    return voices

if __name__ == "__main__":
    d = np.load("tmp_rec.npz")

    x_rec = d["rec"]
    # decode 1 measure at a time...
    x_rec = x_rec[1][None]
    x_rec = np.concatenate([x_rec_i for x_rec_i in x_rec], axis=1)[None]
    measure = x_rec[0, :, :, 0]
    res = decode_measure(measure)
