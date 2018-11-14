"""Simulation library for reversible proximity ligation of DNA probes."""


from collections import defaultdict
from itertools import cycle
from random import sample
from multiprocessing import Pool, cpu_count
from string import letters, lowercase, digits
from time import time
import networkx
from networkx.drawing.nx_agraph import write_dot
import numpy as np
from scipy.spatial.distance import euclidean
from math import (sin,
                  cos,
                  ceil,
                  pi,
                  log10,
                  atan2,
                 )
from shapely import geometry
import matplotlib
import matplotlib.pyplot as plt


def epoch_to_hash(epoch,
                  resolution=100,
                  hashchars='alphanumeric',
                 ):
    """
    Generate an alphanumeric hash from a Unix epoch. Unix epoch is
    rounded to the nearest second before hashing.
    Arguments:
        epoch: Unix epoch time. Must be positive.
    Returns:
        Alphanumeric hash of the Unix epoch time.
    Cribbed & modified from Scott W Harden's website
    http://www.swharden.com/blog/2014-04-19-epoch-timestamp-hashing/
    """
    if epoch <= 0:
        raise ValueError("epoch must be positive.")
    epoch = round(epoch * resolution)
    if hashchars == 'original':
        hashchars = digits + lowercase
    elif hashchars == 'binary':
        hashchars = '01'
    elif hashchars == 'alphanumeric':
        hashchars = letters + digits
    else:
        raise ValueError("Invalid hashchars option.")
    epoch_hash = ''
    while epoch > 0:
        epoch_hash = hashchars[int(epoch % len(hashchars))] + epoch_hash
        epoch = int(epoch / len(hashchars))
    return epoch_hash


def hash_to_epoch(epoch_hash,
                  resolution=100,
                  hashchars='alphanumeric',
                 ):
    """
    Invert hashing function _epoch_to_hash.
    Arguments:
        epoch_hash: Alphanumeric hash of Unix epoch time as returned by
            _epoch_to_hash.
    Returns:
        epoch: Unix epoch time corresponding to epoch_hash.
    Cribbed & modified from Scott W Harden's website
    http://www.swharden.com/blog/2014-04-19-epoch-timestamp-hashing/
    """
    if hashchars == 'original':
        hashchars = digits + lowercase
    elif hashchars == 'binary':
        hashchars = '01'
    elif hashchars == 'alphanumeric':
        hashchars = letters + digits
    else:
        raise ValueError("Invalid hashchars option.")
    #reverse character order
    epoch_hash = epoch_hash[::-1]
    epoch = 0
    for i, c in enumerate(epoch_hash):
        if c not in hashchars:
            raise ValueError("epoch_hash contains unrecognized character(s).")
        epoch += hashchars.find(c)*(len(hashchars)**i)
    return float(epoch) / resolution


def generate_epoch_hash(silent=True):
    epoch_hash = epoch_to_hash(time())
    if not silent:
        print("Generated hash: " + str(epoch_hash))
    return epoch_hash


def square_poisson(size,
                   lam,
                  ):
    mean_num_points = lam * size**2
    num_points = np.random.poisson(lam=mean_num_points)
    x_coordinates = np.random.uniform(low=0.0,
                                      high=size,
                                      size=num_points,
                                     )
    y_coordinates = np.random.uniform(low=0.0,
                                      high=size,
                                      size=num_points,
                                     )
    coordinates = np.stack([x_coordinates, y_coordinates], axis=-1)
    coordinates_list = coordinates.tolist()
    polarities = np.random.choice([3, 5], size=num_points)
    polarities_list = polarities.tolist()
    probes_tuple = tuple([(x, y, polarity)
                          for (x, y), polarity in zip(coordinates_list, polarities_list)])
    return probes_tuple

def shift_probes(probe_coordinates,
                 x_shift,
                 y_shift,
                ):
    return tuple([(x + x_shift, y + y_shift, p) for (x, y, p) in probe_coordinates])

def poisson_squares(square_lattice,
                    size,
                    lam,
                   ):
    probe_coordinates = []
    for x, y in square_lattice:
        square = square_poisson(size=size,
                                lam=lam,
                               )
        square = shift_probes(square, x * size, y * size)
        probe_coordinates += list(square)
    return tuple(probe_coordinates)

def letter_poisson(letter,
                   size,
                   lam,
                  ):
    if letter == 'L':
        square_1 = square_poisson(size=size,
                                  lam=lam,
                                 )
        square_2 = square_poisson(size=size,
                                  lam=lam,
                                 )
        square_2 = shift_probes(square_2, size, 0)
        square_3 = square_poisson(size=size,
                                  lam=lam,
                                 )
        square_3 = shift_probes(square_3, 2 * size, 0)
        square_4 = square_poisson(size=size,
                                  lam=lam,
                                 )
        square_4 = shift_probes(square_3, 0, size)
        probe_coordinates = sum([square_1, square_2, square_3, square_4], ())
    elif letter == 'H':
        square_1 = square_poisson(size=size,
                                  lam=lam,
                                 )
        square_2 = square_poisson(size=size,
                                  lam=lam,
                                 )
        square_2 = shift_probes(square_2, size, 0)
        square_3 = square_poisson(size=size,
                                  lam=lam,
                                 )
        square_3 = shift_probes(square_3, 2 * size, 0)
        square_4 = square_poisson(size=size,
                                  lam=lam,
                                 )
        square_4 = shift_probes(square_4, size, size)
        square_5 = square_poisson(size=size,
                                  lam=lam,
                                 )
        square_5 = shift_probes(square_5, 0, 2 * size)
        square_6 = square_poisson(size=size,
                                  lam=lam,
                                 )
        square_6 = shift_probes(square_6, size, 2 * size)
        square_7 = square_poisson(size=size,
                                  lam=lam,
                                 )
        square_7 = shift_probes(square_7, 2 * size, 2 * size)
        probe_coordinates = sum([square_1,
                                 square_2,
                                 square_3,
                                 square_4,
                                 square_5,
                                 square_6,
                                 square_7,
                                ], ())
    elif letter == 'E':
        letter_lattice = ((0, 0),
                          (1, 0),
                          (2, 0),
                          (3, 0),
                          (4, 0), #This and above are backbone
                          (0, 1),
                          (0, 2), #These two are bottom wing
                          (4, 1),
                          (4, 2), #These two are top wing
                          (2, 1), #This is middle wing
                         )
        probe_coordinates = poisson_squares(square_lattice=letter_lattice,
                                            size=size,
                                            lam=lam,
                                           )
    elif letter == 'D':
        letter_lattice = ((0, 0),
                          (1, 0),
                          (2, 0),
                          (3, 0),
                          (4, 0),
                          (0, 1),
                          (0, 2),
                          (4, 1),
                          (4, 2),
                          (1, 2),
                          (1, 3),
                          (3, 2),
                          (3, 3),
                          (2, 3),
                         )
        probe_coordinates = poisson_squares(square_lattice=letter_lattice,
                                            size=size,
                                            lam=lam,
                                           )
    elif letter == 'W':
        letter_lattice = ((2, 0),
                          (1, 0),
                          (0, 0),
                          (2, 1),
                          (3, 1),
                          (3, 2),
                          (3, 3),
                          (2, 3),
                          (1, 3),
                          (3, 4),
                          (3, 5),
                          (2, 5),
                          (2, 6),
                          (1, 6),
                          (0, 6),
                         )
        probe_coordinates = poisson_squares(square_lattice=letter_lattice,
                                            size=size,
                                            lam=lam,
                                           )
    elif letter == 'R':
        letter_lattice = ((0, 0),
                          (1, 0),
                          (2, 0),
                          (0, 1),
                          (0, 2),
                          (1, 2),
                          (2, 1),
                          (2, 2),
                          (3, 0),
                          (4, 0),
                          (3, 2),
                          (3, 3),
                          (4, 3),
                         )
        probe_coordinates = poisson_squares(square_lattice=letter_lattice,
                                            size=size,
                                            lam=lam,
                                           )
    elif letter == 'O':
        letter_lattice = ((0, 0),
                          (1, 0),
                          (2, 0),
                          (0, 1),
                          (0, 2),
                          (1, 2),
                          (2, 1),
                          (2, 2),
                         )
        probe_coordinates = poisson_squares(square_lattice=letter_lattice,
                                            size=size,
                                            lam=lam,
                                           )
    else:
        raise NotImplementedError("Letter not available.")
    return probe_coordinates

def ligate_graph(probe_graph,
                 max_ligation_distance=1,
                ):
    raise DeprecationWarning("Use rapid_ligate_graph.")
    for (x1, y1, p1) in probe_graph:
        for (x2, y2, p2) in probe_graph:
            if p1 == p2:
                continue
            probe_distance = euclidean((x1, y1), (x2, y2))
            if probe_distance <= max_ligation_distance:
                probe_graph.add_edge((x1, y1, p1), (x2, y2, p2))

def add_line(array,
             ix1, iy1, ix2, iy2,
             color=[0, 0, 255],
            ):
    if ix1 == ix2:
        stepdir = 1 if iy1 < iy2 else -1
        for y in range(iy1 + 1, iy2, stepdir):
            array[ix1, y] = color
    elif iy1 == iy2:
        stepdir = 1 if ix1 < ix2 else -1
        for x in range(ix1 + 1, ix2, stepdir):
            array[x, iy1] = color
    else:
        slope = float(iy2 - iy1) / (ix2 - ix1)
        stepdir = 1 if ix1 < ix2 else -1
        for x in range(ix1 + 1, ix2, stepdir):
            y = int(round(slope * (x - ix1) + iy1))
            if not min(iy1, iy2) <= y <= max(iy1, iy2):
                continue
            array[x, y] = color

        slope = float(ix2 - ix1) / (iy2 - iy1)
        stepdir = 1 if iy1 < iy2 else -1
        for y in range(iy1 + 1, iy2, stepdir):
            x = int(round(slope * (y - iy1) + ix1))
            if not min(ix1, ix2) <= x <= max(ix1, ix2):
                continue
            array[x, y] = color

def squares_tesselate(min_x, max_x,
                      min_y, max_y,
                      size,
                     ):
    if min_x > max_x:
        raise ValueError("min_x = " + str(min_x) + " > max_x = " + str(max_x))
    if min_y > max_y:
        raise ValueError("min_y = " + str(min_y) + " > max_y = " + str(max_y))
    if size <= 0:
        raise ValueError("size = " + str(size) + " <= 0")
    squares = []
    #make offset squares
    start_x, start_y = min_x - size / 2.0, min_y - size / 2.0
    end_x, end_y = max_x + size / 2.0, max_y + size / 2.0
    num_x = int(ceil(float(end_x - start_x) / size))
    num_y = int(ceil(float(end_y - start_y) / size))
    for i in range(num_x):
        for j in range(num_y):
            square_start_x = start_x + i * size
            square_start_y = start_y + j * size
            square_end_x = start_x + (i + 1) * size
            square_end_y = start_y + (j + 1) * size
            squares.append((square_start_x,
                            square_start_y,
                            square_end_x,
                            square_end_y,
                           ))
    #broadest square coverage
    start_x, start_y = min_x - size, min_y - size
    end_x, end_y = max_x + size, max_y + size
    num_x = int(ceil(float(end_x - start_x) / size))
    num_y = int(ceil(float(end_y - start_y) / size))
    for i in range(num_x):
        for j in range(num_y):
            square_start_x = start_x + i * size
            square_start_y = start_y + j * size
            square_end_x = start_x + (i + 1) * size
            square_end_y = start_y + (j + 1) * size
            squares.append((square_start_x,
                            square_start_y,
                            square_end_x,
                            square_end_y,
                           ))
    return tuple(squares)

def rapid_ligate_graph(probe_graph,
                       max_ligation_distance=1,
                       remove_isolates=True,
                      ):
    graph_x_coords = [x for (x, y, p) in probe_graph]
    graph_y_coords = [y for (x, y, p) in probe_graph]
    min_x, max_x = min(graph_x_coords), max(graph_x_coords)
    min_y, max_y = min(graph_y_coords), max(graph_y_coords)
    squares = squares_tesselate(min_x, max_x,
                                min_y, max_y,
                                max_ligation_distance * 2,
                               )
    per_square_nodes = defaultdict(list)
    all_nodes_check_1, all_nodes_check_2 = set(), set()

    for (x, y, p) in probe_graph:
        all_nodes_check_1.add((x, y, p))
        for square in squares:
            sq_start_x, sq_start_y, sq_stop_x, sq_stop_y = square
            if sq_start_x <= x <= sq_stop_x and sq_start_y <= y <= sq_stop_y:
                per_square_nodes[square].append((x, y, p))
                all_nodes_check_2.add((x, y, p))

    assert all_nodes_check_1 == all_nodes_check_2

    for square, nodes in per_square_nodes.iteritems():
        for node1 in nodes:
            x1, y1, p1 = node1
            for node2 in nodes:
                x2, y2, p2 = node2
                if p1 == p2:
                    continue
                #p1 == p2 implies this check:
                #if node1 == node2:
                #    continue
                probe_distance = euclidean((x1, y1), (x2, y2))
                if probe_distance <= max_ligation_distance:
                    probe_graph.add_edge((x1, y1, p1), (x2, y2, p2))
    if remove_isolates:
        subgraphs = networkx.connected_component_subgraphs(probe_graph)
        largest_subgraph = max(subgraphs,
                               key=networkx.number_of_nodes,
                              )
        probe_graph = largest_subgraph
    return probe_graph

def plot_probes(probe_graph,
                five_prime_color=[255, 0, 0],
                three_prime_color=[255, 255, 0],
                connection_color=[0, 0, 255],
                highlight_color=[255, 255, 0],
                expansion=10,
                figsize=15,
                include_nodes=None,
                highlight_edges=None,
               ):
    if include_nodes is not None:
        include_nodes = set(include_nodes)
    if highlight_edges is not None:
        highlight_edges = set(highlight_edges)
    all_x_coordinates = [x for (x, y, polarity) in probe_graph]
    min_x, max_x = min(all_x_coordinates), max(all_x_coordinates)
    all_y_coordinates = [y for (x, y, polarity) in probe_graph]
    min_y, max_y = min(all_y_coordinates), max(all_y_coordinates)
    x_plot_shift = -min_x if min_x < 0 else 0
    y_plot_shift = -min_y if min_y < 0 else 0
    plot_size_x = int(ceil(expansion * (max_x - min_x + 2)))
    plot_size_y = int(ceil(expansion * (max_y - min_y + 2)))

    display_array = np.zeros((plot_size_x, plot_size_y, 3), dtype=np.uint8)

    for (x1, y1, p1), (x2, y2, p2) in probe_graph.edges():
        if include_nodes is not None and ((x1, y1, p1) not in include_nodes
                                          or (x2, y2, p2) not in include_nodes):
            continue
        ix1 = int(round((x1 + x_plot_shift) * expansion))
        iy1 = int(round((y1 + y_plot_shift) * expansion))
        ix2 = int(round((x2 + x_plot_shift) * expansion))
        iy2 = int(round((y2 + y_plot_shift) * expansion))
        if highlight_edges is not None and (((x1, y1, p1), (x2, y2, p2)) in highlight_edges
                                            or ((x2, y2, p2), (x1, y1, p1)) in highlight_edges
                                           ):
            line_color = highlight_color
        else:
            line_color = connection_color
        add_line(display_array,
                 ix1, iy1, ix2, iy2,
                 line_color,
                )

    for (x, y, p) in probe_graph.nodes():
        if include_nodes is not None and (x, y, p) not in include_nodes:
            continue
        ix = int(round((x + x_plot_shift) * expansion))
        iy = int(round((y + y_plot_shift) * expansion))
        node_color = five_prime_color if p == 5 else three_prime_color
        display_array[ix, iy] = node_color

    fig = plt.figure(figsize=(figsize, figsize))
    plt.imshow(display_array)

def polsby_popper(area, perimeter):
    return 4.0 * pi * area / perimeter**2

def graph_polsby_popper(graph):
    points = geometry.MultiPoint([(x, y) for (x, y, p) in graph.nodes()])
    convex_hull = points.convex_hull
    area = convex_hull.area
    perimeter = convex_hull.length
    pp = polsby_popper(area, perimeter)
    return pp

def compare_positions(probe_graph,
                      central_node,
                      radius,
                      neato_map,
                     ):
    """neato_map is {node = (x, y, p): (neato_x, neato_y)}"""
    subgraph = networkx.ego_graph(G=probe_graph,
                                  n=central_node,
                                  radius=radius,
                                  center=True,
                                 )
    central_x, central_y, central_p = central_node
    probe_distances = {(x, y, p): euclidean((x, y), (central_x, central_y))
                       for (x, y, p) in subgraph.nodes()
                       if (x, y, p) != central_node}
    #use nearest probe for theta
    nearest_node = min(probe_distances, key=probe_distances.get)
    if nearest_node == central_node:
        raise RuntimeError("Nearest node is central node.")
    elif probe_distances[nearest_node] == 0:
        raise RuntimeError("Nearest node on top of central node.")
    nearest_x, nearest_y, nearest_p = nearest_node
    theta = atan2(nearest_y - central_y, nearest_x - central_x)
    central_neato_x, central_neato_y = neato_map[central_node]
    nearest_neato_x, nearest_neato_y = neato_map[nearest_node]
    neato_theta = atan2(nearest_neato_y - central_neato_y,
                        nearest_neato_x - central_neato_x,
                       )
    discrepancy_ledger = {}
    for node in subgraph.nodes():
        if node == central_node:
            continue
        neato_x, neato_y = neato_map[node]
        neato_distance = euclidean((central_neato_x, central_neato_y), (neato_x, neato_y))
        node_neato_theta = atan2(neato_y - central_neato_y, neato_x - central_neato_x)
        node_neato_theta_diff = atan2(sin(node_neato_theta - neato_theta),
                                      cos(node_neato_theta - neato_theta),
                                     )
        x, y, p = node
        node_theta = atan2(y - central_y, x - central_x)
        node_theta_diff = atan2(sin(node_theta - theta), cos(node_theta - theta))
        node_distance = probe_distances[node]
        assert node not in discrepancy_ledger
        discrepancy_ledger[node] = (node_distance,
                                    node_theta_diff,
                                    neato_distance,
                                    node_neato_theta_diff,
                                   )
    return discrepancy_ledger

def node_angle(central_node,
               node_1,
               node_2,
              ):
    central_x, central_y = central_node[:2]
    x1, y1 = node_1[:2]
    x2, y2 = node_2[:2]
    theta_1 = atan2(y1 - central_y, x1 - central_x)
    theta_2 = atan2(y2 - central_y, x2 - central_x)
    return atan2(sin(theta_1 - theta_2), cos(theta_1 - theta_2))

def ld_1(s, t):
    if not s: return len(t)
    if not t: return len(s)
    if s[0] == t[0]: return ld(s[1:], t[1:])
    l1 = ld(s, t[1:])
    l2 = ld(s[1:], t)
    l3 = ld(s[1:], t[1:])
    return 1 + min(l1, l2, l3)

def ld_2(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and
                                                 #current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def ld_3(s, t):
    ''' From Wikipedia article; Iterative with two matrix rows. '''
    if s == t: return 0
    elif len(s) == 0: return len(t)
    elif len(t) == 0: return len(s)
    v0 = [None] * (len(t) + 1)
    v1 = [None] * (len(t) + 1)
    for i in range(len(v0)):
        v0[i] = i
    for i in range(len(s)):
        v1[0] = i + 1
        for j in range(len(t)):
            cost = 0 if s[i] == t[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        for j in range(len(v0)):
            v0[j] = v1[j]
    return v1[len(t)]

def levenshtein_comparison(probe_graph,
                           central_node,
                           radius,
                           neato_map,
                           ld_algorithm=3,
                           normalize=False,
                           bidirectional=False,
                          ):
    subgraph = networkx.ego_graph(G=probe_graph,
                                  n=central_node,
                                  radius=radius,
                                  center=False,
                                 )
    central_x, central_y, central_p = central_node
    probe_distances = {(x, y, p): euclidean((x, y), (central_x, central_y))
                       for (x, y, p) in subgraph.nodes()
                       if (x, y, p) != central_node}
    #use nearest probe for baseline
    nearest_node = min(probe_distances, key=probe_distances.get)
    if nearest_node == central_node:
        raise RuntimeError("Nearest node is central node.")
    elif probe_distances[nearest_node] == 0:
        raise RuntimeError("Nearest node on top of central node.")
    node_angles = []
    neato_angles = []
    for node in subgraph:
        angle = node_angle(central_node=central_node,
                           node_1=nearest_node,
                           node_2=node,
                          )
        node_angles.append(angle)
        neato_angle = node_angle(central_node=neato_map[central_node],
                                 node_1=neato_map[nearest_node],
                                 node_2=neato_map[node]
                                )
        neato_angles.append(neato_angle)
    sorted_node_angles = sorted(enumerate(node_angles), key=lambda x:x[1])
    sorted_node_indexes = [i for i, a in sorted_node_angles]
    sorted_neato_angles = sorted(enumerate(neato_angles), key=lambda x:x[1])
    sorted_neato_indexes = [i for i, a in sorted_neato_angles]
    assert len(sorted_node_indexes) == len(sorted_neato_indexes)
    if ld_algorithm == 1:
        ld = ld_1
    elif ld_algorithm == 2:
        ld = ld_2
    elif ld_algorithm == 3:
        ld = ld_3
    else:
        raise ValueError("Invalid ld_algorithm.")
    distance = ld(sorted_node_indexes, sorted_neato_indexes)
    if bidirectional:
        sorted_neato_indexes.reverse()
        r_distance = ld(sorted_node_indexes, sorted_neato_indexes)
        distance = min(distance, r_distance)
    if normalize:
        distance /= float(len(sorted_node_indexes))
    return distance

def kendalltau_comparison(probe_graph,
                          central_node,
                          radius,
                          neato_map,
                          absolute=False,
                         ):
    subgraph = networkx.ego_graph(G=probe_graph,
                                  n=central_node,
                                  radius=radius,
                                  center=False,
                                 )
    central_x, central_y, central_p = central_node
    probe_distances = {(x, y, p): euclidean((x, y), (central_x, central_y))
                       for (x, y, p) in subgraph.nodes()
                       if (x, y, p) != central_node}
    #use nearest probe for baseline
    nearest_node = min(probe_distances, key=probe_distances.get)
    if nearest_node == central_node:
        raise RuntimeError("Nearest node is central node.")
    elif probe_distances[nearest_node] == 0:
        raise RuntimeError("Nearest node on top of central node.")
    node_angles = []
    neato_angles = []
    for node in subgraph:
        angle = node_angle(central_node=central_node,
                           node_1=nearest_node,
                           node_2=node,
                          )
        node_angles.append(angle)
        neato_angle = node_angle(central_node=neato_map[central_node],
                                 node_1=neato_map[nearest_node],
                                 node_2=neato_map[node]
                                )
        neato_angles.append(neato_angle)
    distance = kendalltau(node_angles, neato_angles).correlation
    if absolute:
        distance = abs(distance)
    return distance

def rgb_to_hex(R, G, B,
               multiplier=255,
              ):
    R = int(round(R * multiplier))
    G = int(round(G * multiplier))
    B = int(round(B * multiplier))
    return '#%02x%02x%02x' % (R, G, B)

def displacement_distances(probe_graph,
                           central_node,
                           radius,
                           neato_map,
                          ):
    subgraph = networkx.ego_graph(G=probe_graph,
                                  n=central_node,
                                  radius=radius,
                                  center=True,
                                 )
    central_x, central_y, central_p = central_node
    probe_distances = {(x, y, p): euclidean((x, y), (central_x, central_y))
                       for (x, y, p) in subgraph.nodes()
                       if (x, y, p) != central_node}
    #use nearest probe for theta
    nearest_node = min(probe_distances, key=probe_distances.get)
    if nearest_node == central_node:
        raise RuntimeError("Nearest node is central node.")
    elif probe_distances[nearest_node] == 0:
        raise RuntimeError("Nearest node on top of central node.")
    nearest_x, nearest_y, nearest_p = nearest_node
    theta = atan2(nearest_y - central_y, nearest_x - central_x)
    central_neato_x, central_neato_y = neato_map[central_node]
    nearest_neato_x, nearest_neato_y = neato_map[nearest_node]
    neato_theta = atan2(nearest_neato_y - central_neato_y,
                        nearest_neato_x - central_neato_x,
                       )
    reset_rotation = theta - neato_theta
    r_sin, r_cos = sin(reset_rotation), cos(reset_rotation)
    displacement_ledger = {}
    for node in subgraph:
        if node == central_node:
            displacement_ledger[node] = 0
            continue
        x, y, p = node
        x, y = x - central_x, y - central_y
        neato_x, neato_y = neato_map[node]
        neato_x, neato_y = neato_x - central_neato_x, neato_y - central_neato_y
        r_neato_x = neato_x * r_cos - neato_y * r_sin
        r_neato_y = neato_x * r_sin + neato_y * r_cos
        displacement = euclidean((x, y), (r_neato_x, r_neato_y))
        assert node not in displacement_ledger
        displacement_ledger[node] = displacement
    return displacement_ledger

def pairwise_node_distortion(probe_graph,
                             central_node,
                             radius,
                             neato_map,
                            ):
    subgraph = networkx.ego_graph(G=probe_graph,
                                  n=central_node,
                                  radius=radius,
                                  center=True,
                                 )
    pairwise_ledger = {}
    for node_1, node_2 in MCsimlib._pairwise(subgraph.nodes()):
        x1, y1, p1 = node_1
        x2, y2, p2 = node_2
        edge_length = euclidean((x1, y1), (x2, y2))
        neato_x1, neato_y1 = neato_map[node_1]
        neato_x2, neato_y2 = neato_map[node_2]
        neato_edge_length = euclidean((neato_x1, neato_y1), (neato_x2, neato_y2))
        assert (node_1, node_2) not in pairwise_ledger
        pairwise_ledger[(node_1, node_2)] = (edge_length, neato_edge_length)
    return pairwise_ledger

def compare_edge_lengths(probe_graph,
                         central_node,
                         radius,
                         neato_map,
                        ):
    subgraph = networkx.ego_graph(G=probe_graph,
                                  n=central_node,
                                  radius=radius,
                                  center=True,
                                 )
    length_ledger = {}
    for (node_1, node_2) in subgraph.edges():
        x1, y1, p1 = node_1
        x2, y2, p2 = node_2
        edge_length = euclidean((x1, y1), (x2, y2))
        neato_x1, neato_y1 = neato_map[node_1]
        neato_x2, neato_y2 = neato_map[node_2]
        neato_edge_length = euclidean((neato_x1, neato_y1), (neato_x2, neato_y2))
        assert (node_1, node_2) not in length_ledger
        length_ledger[(node_1, node_2)] = (edge_length, neato_edge_length)
    return length_ledger

def pp_ratio(probe_graph,
             central_node,
             radius,
             neato_map,
            ):
    subgraph = networkx.ego_graph(G=probe_graph,
                                  n=central_node,
                                  radius=radius,
                                  center=True,
                                 )
    original_pp = graph_polsby_popper(subgraph)
    subgraph_neato = [neato_map[node] for node in subgraph]
    subgraph_neato_hull = geometry.MultiPoint(subgraph_neato).convex_hull
    neato_area, neato_perimeter = subgraph_neato_hull.area, subgraph_neato_hull.length
    neato_pp = polsby_popper(neato_area, neato_perimeter)
    return float(neato_pp) / float(original_pp)

def area_scaling(probe_graph,
                 central_node,
                 radius,
                 neato_map,
                ):
    subgraph = networkx.ego_graph(G=probe_graph,
                                  n=central_node,
                                  radius=radius,
                                  center=True,
                                 )
    subgraph_hull = geometry.MultiPoint([(x, y) for (x, y, p) in subgraph]).convex_hull
    subgraph_neato = [neato_map[node] for node in subgraph]
    subgraph_neato_hull = geometry.MultiPoint(subgraph_neato).convex_hull
    original_area = subgraph_hull.area
    neato_area = subgraph_neato_hull.area
    return float(original_area) / neato_area

def write_heatmap_dot(input_dot_filename,
                      output_dot_filename,
                      node_values,
                      minimum_value=None,
                      maximum_value=None,
                      heatmap='viridis',
                      per_node_signifier="height",
                      nan_color=(1, 1, 1),
                      blank_nodes=None,
                      blank_color=(1, 1, 1),
                     ):
    if minimum_value is None:
        minimum_value = min(node_values.values())
    if maximum_value is None:
        maximum_value = max(node_values.values())
    output_line_cache = []
    with open(input_dot_filename) as input_dotfile:
        for line in input_dotfile:
            if per_node_signifier not in line:
                output_line_cache.append(line)
                continue
            x_str, y_str, p_str, height_str = line.split(' ')
            x = float(x_str[3:-1])
            y = float(y_str[:-1])
            p = int(p_str[:-3])
            node = x, y, p
            node_value = node_values[node]
            node_value = max(min(node_value, maximum_value), minimum_value)
            heatmap_position = (float(node_value - minimum_value)
                                / (maximum_value - minimum_value)
                               )
            if np.isnan(heatmap_position):
                R, G, B = nan_color
            elif heatmap == 'viridis':
                R, G, B, alpha = matplotlib.cm.viridis(heatmap_position)
            else:
                ValueError("Invalid heatmap.")
            if blank_nodes is not None and node in blank_nodes:
                R, G, B = blank_color
            colorhex = rgb_to_hex(R, G, B)
            fillcolor_str = ('\t\t'
                             + "fillcolor=\""
                             + str(colorhex)
                             + "\"" + ","
                             + '\n'
                            )
            filled_str = "\t\tstyle=filled,\n"
            output_line_cache.append(line)
            output_line_cache.append(filled_str)
            output_line_cache.append(fillcolor_str)
    with open(output_dot_filename, 'w') as output_dotfile:
        output_dotfile.writelines(output_line_cache)

def neato_node_densities(neato_map,
                         radius,
                        ):
    """radius is Euclidean in neato space."""
    pairwise_distances = {}
    for (node_1, (x1, y1)), (node_2, (x2, y2)) in combinations(neato_map.iteritems(), 2):
        if abs(x1 - x2) > radius or abs(y1 - y2) > radius:
            continue
        node_pair = frozenset((node_1, node_2))
        distance = euclidean((x1, y1), (x2, y2))
        pairwise_distances[node_pair] = distance
    #account for each node itself within its local density
    densities = defaultdict(lambda:1)
    for node_pair, distance in pairwise_distances.iteritems():
        node_1, node_2 = tuple(node_pair)
        if distance <= radius:
            densities[node_1] += 1
            densities[node_2] += 1
    return densities

def ligate_additional_nodes(probe_graph,
                            sprinkled_nodes,
                            max_ligation_distance,
                            remove_isolates=True,
                            self_long=False,
                           ):
    augmented_probe_graph = probe_graph.copy()
    for node in augmented_probe_graph.nodes():
        x, y, p = node
        for sprinkled_node in sprinkled_nodes:
            sx, sy, sp = sprinkled_node
            if p == sp:
                continue
            if abs(sx - x) > max_ligation_distance or abs(sy - y) > max_ligation_distance:
                continue
            probe_distance = euclidean((x, y), (sx, sy))
            if probe_distance <= max_ligation_distance:
                augmented_probe_graph.add_edge(node, sprinkled_node, len=max_ligation_distance)
    if self_long:
        self_long_distance = 2 * (max_ligation_distance - 1)
        for (node_1, node_2) in combinations(augmented_probe_graph.nodes(), 2):
            x1, y1, p1 = node_1
            x2, y2, p2 = node_2
            if p1 == p2:
                continue
            if abs(x1 - x2) > self_long_distance or abs(y1 - y2) > self_long_distance:
                continue
            probe_distance = euclidean((x1, y1), (x2, y2))
            if probe_distance <= self_long_distance:
                augmented_probe_graph.add_edge(node_1, node_2, len=self_long_distance)
    if remove_isolates:
        subgraphs = networkx.connected_component_subgraphs(augmented_probe_graph)
        largest_subgraph = max(subgraphs,
                               key=networkx.number_of_nodes,
                              )
        augmented_probe_graph = largest_subgraph
    return augmented_probe_graph

def make_heatmap_colorbar(node_values,
                          minimum_value=None,
                          maximum_value=None,
                          silent=False,
                         ):
    min_value, max_value = min(node_values.values()), max(node_values.values())
    if minimum_value is not None:
        min_value = minimum_value
    if maximum_value is not None:
        max_value = maximum_value
    mean_value = np.mean(node_values.values())
    median_value = np.median(node_values.values())
    data = np.zeros((2, 2))
    data[0, 0] = min_value
    data[1, 1] = max_value
    data[0, 1] = mean_value
    data[1, 0] = median_value
    if not silent:
        print("make_heatmap_colorbar: "
              + "(min_value, mean_value, median_value, max_value) = "
              + str((min_value, mean_value, median_value, max_value))
             )
    heatmap = plt.pcolor(data)
    plt.colorbar(heatmap)
    plt.show()

def neighbor_angle_sequence(probe_graph,
                            central_node,
                            radius,
                            neato_coordinates,
                           ):
    subgraph = networkx.ego_graph(G=probe_graph,
                                  n=central_node,
                                  radius=radius,
                                  center=False,
                                 )
    n_central_x, n_central_y = neato_coordinates[central_node]
    neato_distances = {node: euclidean((n_central_x, n_central_y), neato_coordinates[node])
                       for node in subgraph.nodes()}
    #use nearest probe for baseline
    nearest_node = min(neato_distances, key=neato_distances.get)
    if nearest_node == central_node:
        raise RuntimeError("Nearest node is central node.")
    elif neato_distances[nearest_node] == 0:
        raise RuntimeError("Nearest node on top of central node.")
    neato_angles = {}
    for node in subgraph:
        neato_angle = node_angle(central_node=neato_coordinates[central_node],
                                 node_1=neato_coordinates[nearest_node],
                                 node_2=neato_coordinates[node]
                                )
        neato_angles[node] = neato_angle
    return neato_angles


def read_probe_graph_plain(neato_plain_filename):
    neato_coordinates, graph_scale, graph_width, graph_height = {}, 0, 0, 0
    with open(neato_plain_filename) as neato_plain_file:
        for L, line in enumerate(neato_plain_file, start=1):
            split_line = line.split(' ')
            line_type = split_line[0]
            if line_type == 'graph':
                type_string, graph_scale_str, graph_width_str, graph_height_str = split_line
                graph_scale = float(graph_scale_str)
                graph_width = float(graph_width_str)
                graph_height = float(graph_height_str)
            elif line_type == 'node':
                (type_sting,
                 node_x_str,
                 node_y_str,
                 node_p_str,
                 x_str,
                 y_str,
                 width_str,
                 height_str,
                 label,
                 style,
                 shape,
                 color,
                 fillcolor,
                ) = split_line
                node_x = float(node_x_str[2:-1])
                node_y = float(node_y_str[:-1])
                node_p = int(node_p_str[:-2])
                node = (node_x, node_y, node_p)
                graph_x = float(x_str)
                graph_y = float(y_str)
                assert node not in neato_coordinates
                neato_coordinates[node] = (graph_x, graph_y)
            elif line_type == 'edge':
                pass
            elif line_type == 'stop\n':
                pass
            else:
                raise RuntimeError("Invalid line type. Line #" + str(L) + ": " + str(line))
    return neato_coordinates, graph_scale, graph_width, graph_height

def remove_isolates(probe_graph,
                    silent=False,
                   ):
    subgraphs = networkx.connected_component_subgraphs(probe_graph)
    largest_subgraph = max(subgraphs,
                           key=networkx.number_of_nodes,
                          )
    o_num_nodes, o_num_edges = probe_graph.number_of_nodes(), probe_graph.number_of_edges()
    f_num_nodes, f_num_edges = largest_subgraph.number_of_nodes(), largest_subgraph.number_of_edges()
    if not silent:
        print("Original graph stats: "
              + str(o_num_nodes) + " nodes; "
              + str(o_num_edges) + " edges."
             )
        print("Filtered graph stats: "
              + str(f_num_nodes) + " nodes; "
              + str(f_num_edges) + " edges."
             )
        print(str(o_num_nodes - f_num_nodes)
              + " nodes removed and "
              + str(o_num_edges - f_num_edges)
              + " edges removed."
             )
    return largest_subgraph

def write_probe_graph_dot(probe_graph,
                          base_filename,
                          epoch_hash=None,
                          remove_isolates=False,
                          print_stats=True,
                          print_hash=True,
                          additional_node_attributes=None,
                          additional_edge_attributes=None,
                         ):
    dot_probe_graph = networkx.Graph()
    dot_probe_graph.add_edges_from(probe_graph.edges())

    for node in dot_probe_graph:
        dot_probe_graph.node[node]['shape'] = 'circle'

    if additional_node_attributes is not None:
        for attribute in iter(additional_node_attributes):
            for node in probe_graph:
                if attribute in probe_graph.node[node]:
                    dot_probe_graph.node[node][attribute] = probe_graph.node[node][attribute]

    if additional_edge_attributes is not None:
        for attribute in iter(additional_edge_attributes):
            for (node_1, node_2) in probe_graph.edges():
                if attribute in probe_graph[node_1][node_2]:
                    dot_probe_graph[node_1][node_2][attribute] = probe_graph[node_1][node_2][attribute]

    if remove_isolates:
        subgraphs = networkx.connected_component_subgraphs(dot_probe_graph)
        largest_subgraph = max(subgraphs,
                               key=networkx.number_of_nodes,
                              )
        dot_probe_graph = largest_subgraph

    if print_stats:
        print(str(dot_probe_graph.number_of_nodes()) + " nodes; "
              + str(dot_probe_graph.number_of_edges()) + " edges."
             )

    if epoch_hash is None:
        epoch_hash = generate_hash(silent=(not print_hash))
    filename = base_filename + "." + epoch_hash + ".dot"
    write_dot(dot_probe_graph, filename)
    return filename, epoch_hash

def epoch_to_hash(epoch,
                  resolution=100,
                  hashchars='alphanumeric',
                 ):
    """
    Generate an alphanumeric hash from a Unix epoch. Unix epoch is
    rounded to the nearest second before hashing.
    Arguments:
        epoch: Unix epoch time. Must be positive.
    Returns:
        Alphanumeric hash of the Unix epoch time.
    Cribbed & modified from Scott W Harden's website
    http://www.swharden.com/blog/2014-04-19-epoch-timestamp-hashing/
    """
    if epoch <= 0:
        raise ValueError("epoch must be positive.")
    epoch = round(epoch * resolution)
    if hashchars == 'original':
        hashchars = '0123456789abcdefghijklmnopqrstuvwxyz'
    elif hashchars == 'binary':
        hashchars = '01'
    elif hashchars == 'alphanumeric':
        hashchars = letters + digits
    else:
        raise ValueError("Invalid hashchars option.")
    epoch_hash = ''
    while epoch > 0:
        epoch_hash = hashchars[int(epoch % len(hashchars))] + epoch_hash
        epoch = int(epoch / len(hashchars))
    return epoch_hash


def hash_to_epoch(epoch_hash,
                  resolution=100,
                  hashchars='alphanumeric',
                 ):
    """
    Invert hashing function _epoch_to_hash.
    Arguments:
        epoch_hash: Alphanumeric hash of Unix epoch time as returned by
            _epoch_to_hash.
    Returns:
        epoch: Unix epoch time corresponding to epoch_hash.
    Cribbed & modified from Scott W Harden's website
    http://www.swharden.com/blog/2014-04-19-epoch-timestamp-hashing/
    """
    if hashchars == 'original':
        hashchars = '0123456789abcdefghijklmnopqrstuvwxyz'
    elif hashchars == 'binary':
        hashchars = '01'
    elif hashchars == 'alphanumeric':
        hashchars = letters + digits
    else:
        raise ValueError("Invalid hashchars option.")
    #reverse character order
    epoch_hash = epoch_hash[::-1]
    epoch = 0
    for i, c in enumerate(epoch_hash):
        if c not in hashchars:
            raise ValueError("epoch_hash contains unrecognized character(s).")
        epoch += hashchars.find(c)*(len(hashchars)**i)
    return float(epoch) / resolution

def generate_hash(silent=False):
    epoch_hash = epoch_to_hash(time())
    if not silent:
        print("Generated hash: " + str(epoch_hash))
    return epoch_hash

def zfill_width(num_files):
    return int(int(ceil(log10(num_files))))

def furthest_pair(points):
    if len(points) < 2:
        raise ValueError("Need at least a pair of points; points = " + str(points))
    elif len(points) == 2:
        p1, p2 = points
    else:
        convex_hull = geometry.MultiPoint(points).convex_hull
        pairwise_distances = {(p1, p2): euclidean(p1, p2)
                              for p1, p2 in combinations(convex_hull.exterior.coords, 2)}
        p1, p2 = max(pairwise_distances, key=pairwise_distances.get)
    return p1, p2

def rotate(x, y, r):
    #separate line because sin & cos are expensive
    r_sin, r_cos = sin(r), cos(r)
    return x * r_cos - y * r_sin, x * r_sin + y * r_cos

def quick_align(graph_A_neato_coordinates,
                graph_B_neato_coordinates,
               ):
    overlapping_nodes = set(graph_A_neato_coordinates) & set(graph_B_neato_coordinates)
    if len(overlapping_nodes) < 2:
        raise ValueError("Graphs must share at least two nodes.")
    if len(set(graph_A_neato_coordinates.values())) < len(graph_A_neato_coordinates.values()):
        raise RuntimeError("Graph A node coordinates not unique.")
    if len(set(graph_B_neato_coordinates.values())) < len(graph_B_neato_coordinates):
        raise RuntimeError("Graph B node coordinates not unique.")
    inverse_graph_A_neato_coordinates = {(x, y): node
                                         for node, (x, y)
                                         in graph_A_neato_coordinates.iteritems()}
    inverse_graph_B_neato_coordinates = {(x, y): node
                                         for node, (x, y)
                                         in graph_B_neato_coordinates.iteritems()}
    overlapping_A_coordinates = {node: graph_A_neato_coordinates[node]
                                 for node in list(overlapping_nodes)}
    (Afx1, Afy1), (Afx2, Afy2) = furthest_pair(overlapping_A_coordinates.values())
    anchor_1 = inverse_graph_A_neato_coordinates[(Afx1, Afy1)]
    anchor_2 = inverse_graph_A_neato_coordinates[(Afx2, Afy2)]
    (Bfx1, Bfy1), (Bfx2, Bfy2) = graph_B_neato_coordinates[anchor_1], graph_B_neato_coordinates[anchor_2]
    Afx_centroid, Afy_centroid = np.mean([Afx1, Afx2]), np.mean([Afy1, Afy2])
    Bfx_centroid, Bfy_centroid = np.mean([Bfx1, Bfx2]), np.mean([Bfy1, Bfy2])
    Btranslate_x, Btranslate_y = Afx_centroid - Bfx_centroid, Afy_centroid - Bfy_centroid
    translated_B = {node: (x + Btranslate_x, y + Btranslate_y)
                    for node, (x, y) in graph_B_neato_coordinates.iteritems()}
    A_anchor_angle = node_angle((0, 0),
                                (0, 1),
                                (Afx1 - Afx_centroid, Afy1 - Afy_centroid),
                               )
    B_anchor_angle = node_angle((0, 0),
                                (0, 1),
                                (Bfx1 - Bfx_centroid, Bfy1 - Bfy_centroid),
                               )
    r_sin, r_cos = sin(A_anchor_angle - B_anchor_angle), cos(A_anchor_angle - B_anchor_angle)
    aligned_B = {node: (x * r_cos - y * r_sin, x * r_sin + y * r_cos)
                 for node, (x, y) in translated_B.iteritems()}
    return aligned_B

def matching_orientation(graph_A_neato_coordinates,
                         graph_B_neato_coordinates,
                        ):
    overlapping_nodes = set(graph_A_neato_coordinates) & set(graph_B_neato_coordinates)
    if len(overlapping_nodes) < 2:
        raise ValueError("Graphs must share at least two nodes.")
    aligned_B = quick_align(graph_A_neato_coordinates,
                            graph_B_neato_coordinates,
                           )
    distance_match = sum([euclidean(Bcoords, graph_A_neato_coordinates[node])
                          for node, Bcoords in aligned_B.iteritems()
                          if node in overlapping_nodes])
    mirrored_B = {node: (-x, y)
                  for node, (x, y) in graph_B_neato_coordinates.iteritems()}
    mirror_aligned_B = quick_align(graph_A_neato_coordinates,
                                   mirrored_B,
                                  )
    mirror_distance_match = sum([euclidean(Bcoords, graph_A_neato_coordinates[node])
                                 for node, Bcoords in mirror_aligned_B.iteritems()
                                 if node in overlapping_nodes])
    if distance_match <= mirror_distance_match:
        matches = True
    else:
        matches = False
    return matches, distance_match, mirror_distance_match, len(overlapping_nodes)

def find_most_redundant_neighborhood(graph_cover,
                                     minimum_coverage=1,
                                    ):
    if minimum_coverage < 1:
        raise RuntimeWarning("miniumum_coverage " + str(minimum_coverage) + " may leave uncovered nodes.")
    node_coverages = defaultdict(int)
    for neighborhood in graph_cover.itervalues():
        for node in neighborhood:
            node_coverages[node] += 1
    neighborhood_redundancies = {}
    for central_node, neighborhood in graph_cover.iteritems():
        neighborhood_coverages = [node_coverages[node] for node in neighborhood]
        if min(neighborhood_coverages) == 0:
            raise Exception("Uncovered node found! " + str((central_node, neighborhood_coverages)))
        elif min(neighborhood_coverages) <= minimum_coverage:
            #cannot remove last cover for a node
            continue
        neighborhood_redundancies[central_node] = np.mean(neighborhood_coverages)
    if neighborhood_redundancies:
        most_redundant_neighborhood = max(neighborhood_redundancies, key=neighborhood_redundancies.get)
    else:
        most_redundant_neighborhood = None
    return most_redundant_neighborhood

def subsampled_graph_cover(full_graph_cover,
                           subsample_rate,
                          ):
    num_subsamples = int(ceil(len(full_graph_cover) / float(subsample_rate)))
    subsampled_nodes = sample(full_graph_cover, num_subsamples)
    attempted_graph_cover = {node: full_graph_cover[node] for node in subsampled_nodes}
    originally_covered_nodes = set([node
                                    for neighborhood in full_graph_cover.itervalues()
                                    for node in neighborhood]
                                  )
    subsample_covered_nodes = set([node
                                   for neighborhood in attempted_graph_cover.itervalues()
                                   for node in neighborhood])
    uncovered_nodes = originally_covered_nodes - subsample_covered_nodes
    required_cover = {node: full_graph_cover[node]
                      for node in list(uncovered_nodes)}
    attempted_graph_cover.update(required_cover)
    return attempted_graph_cover, len(uncovered_nodes)

def round_up_to_even(x):
    return int(ceil(float(x) / 2) * 2)

def get_heatmap_color(value,
                      min_value,
                      max_value,
                      heatmap='viridis',
                      nan_color=(0, 0, 0),
                     ):
    value = max(min(value, max_value), min_value)
    heatmap_position = (float(value - min_value)
                        / (max_value - min_value)
                       )
    if np.isnan(heatmap_position):
        R, G, B = nan_color
    elif heatmap == 'viridis':
        R, G, B, alpha = matplotlib.cm.viridis(heatmap_position)
    else:
        ValueError("Invalid heatmap.")
    colorhex = rgb_to_hex(R, G, B)
    return colorhex


def displacement_distances_v2(probe_graph,
                              true_positions,
                              recovered_positions,
                              central_node,
                              radius,
                             ):
    subgraph = networkx.ego_graph(G=probe_graph,
                                  n=central_node,
                                  radius=radius,
                                  center=True,
                                 )
    probe_distances = {node: euclidean(true_positions[node],
                                       true_positions[central_node],
                                      )
                       for node in subgraph.nodes()
                       if node != central_node}
    #use nearest probe for theta
    nearest_node = min(probe_distances, key=probe_distances.get)
    if nearest_node == central_node:
        raise RuntimeError("Nearest node is central node.")
    elif probe_distances[nearest_node] == 0:
        raise RuntimeError("Nearest node on top of central node.")
    central_x, central_y = true_positions[central_node]
    nearest_x, nearest_y = true_positions[nearest_node]
    theta = atan2(nearest_y - central_y, nearest_x - central_x)
    central_recovered_x, central_recovered_y = recovered_positions[central_node]
    nearest_recovered_x, nearest_recovered_y = recovered_positions[nearest_node]
    recovered_theta = atan2(nearest_recovered_y - central_recovered_y,
                            nearest_recovered_x - central_recovered_x,
                           )
    reset_rotation = theta - recovered_theta
    r_sin, r_cos = sin(reset_rotation), cos(reset_rotation)
    displacement_ledger = {}
    for node in subgraph:
        if node == central_node:
            displacement_ledger[node] = 0
            continue
        x, y = true_positions[node]
        x, y = x - central_x, y - central_y
        recovered_x, recovered_y = recovered_positions[node]
        recovered_x, recovered_y = recovered_x - central_recovered_x, recovered_y - central_recovered_y
        r_recovered_x = recovered_x * r_cos - recovered_y * r_sin
        r_recovered_y = recovered_x * r_sin + recovered_y * r_cos
        displacement = euclidean((x, y), (r_recovered_x, r_recovered_y))
        assert node not in displacement_ledger
        displacement_ledger[node] = displacement
    return displacement_ledger

def all_displacement_distances_v2(probe_graph,
                                  true_positions,
                                  recovered_positions,
                                  radius
                                 ):
    return {node: displacement_distances_v2(probe_graph=probe_graph,
                                            true_positions=true_positions,
                                            recovered_positions=recovered_positions,
                                            central_node=node,
                                            radius=radius,
                                           )
            for node in probe_graph}

def get_best_alignment(probe_graph,
                       true_positions,
                       recovered_positions,
                       radius=9,
                       flip_x=False,
                       flip_y=False,
                       resolution=100,
                      ):
    average_true_position_x = np.mean([x for x, y in true_positions.itervalues()])
    average_true_position_y = np.mean([y for x, y in true_positions.itervalues()])
    anchor_distances = {node: euclidean((x, y), (average_true_position_x, average_true_position_y))
                        for node, (x, y) in true_positions.iteritems()}
    anchor_node = min(anchor_distances, key=anchor_distances.get)
    anchor_x, anchor_y = true_positions[anchor_node]
    anchored_true_positions = {node: (x - anchor_x, y - anchor_y)
                               for node, (x, y) in true_positions.iteritems()}
    recovered_anchor_x, recovered_anchor_y = recovered_positions[anchor_node]
    anchored_recovered_positions = {node: (x - recovered_anchor_x, y - recovered_anchor_y)
                                    for node, (x, y) in recovered_positions.iteritems()}
    if flip_x:
        anchored_recovered_positions = {node: (-x, y)
                                        for node, (x, y) in anchored_recovered_positions.iteritems()}
    if flip_y:
        anchored_recovered_positions = {node: (x, -y)
                                        for node, (x, y) in anchored_recovered_positions.iteritems()}
    subgraph = networkx.ego_graph(G=probe_graph,
                                  n=anchor_node,
                                  radius=radius,
                                  center=True,
                                 )
    alignment_angles = np.radians(np.arange(0, 361, 1.0 / resolution)).tolist()
    alignment_sin = np.sin(alignment_angles).tolist()
    alignment_cos = np.cos(alignment_angles).tolist()
    assert all([sin(angle) == alignment_sin[a] for a, angle in enumerate(alignment_angles)])
    assert all([cos(angle) == alignment_cos[a] for a, angle in enumerate(alignment_angles)])
    assert alignment_angles[-1] <= 360
    if alignment_angles[-1] == 360:
        alignment_angles = alignment_angles[:-1]
        alignment_sin = alignment_sin[:-1]
        alignment_cos = alignment_cos[:-1]
    best_alignment, best_distance = None, None
    for a, angle in enumerate(alignment_angles):
        distance = 0
        r_sin, r_cos = alignment_sin[a], alignment_cos[a]
        for node in subgraph:
            atx, aty = anchored_true_position = anchored_true_positions[node]
            arx, ary = anchored_recovered_position = anchored_recovered_positions[node]
            rarx = arx * r_cos - ary * r_sin
            rary = arx * r_sin + ary * r_cos
            distance += euclidean((atx, aty), (rarx, rary))
        if best_distance is None or distance < best_distance:
            best_alignment, best_distance = angle, distance
    return best_alignment, best_distance, anchor_node

def rotate_graph(positions,
                 anchor_node,
                 angle,
                 flip_x=False,
                 flip_y=False,
                ):
    anchor_x, anchor_y = positions[anchor_node]
    anchored_positions = {node: (x - anchor_x, y - anchor_y)
                          for node, (x, y) in positions.iteritems()}
    if flip_x:
        anchored_positions = {node: (-x, y)
                              for node, (x, y) in anchored_positions.iteritems()}
    if flip_y:
        anchored_positions = {node: (x, -y)
                              for node, (x, y) in anchored_positions.iteritems()}
    r_sin, r_cos = sin(angle), cos(angle)
    rotated_positions = {node: (ax * r_cos - ay * r_sin, ax * r_sin + ay * r_cos)
                         for node, (ax, ay) in anchored_positions.iteritems()}
    reset_positions = {node: (x + anchor_x, y + anchor_y)
                       for node, (x, y) in rotated_positions.iteritems()}
    return reset_positions

def anchor_translate(positions,
                     anchor_node,
                    ):
    anchor_x, anchor_y = positions[anchor_node]
    return {node: (x - anchor_x, y - anchor_y)
            for node, (x, y) in positions.iteritems()}

def node_angle_v2(central_node_xy,
                  node_1_xy,
                  node_2_xy,
                 ):
    central_x, central_y = central_node_xy
    x1, y1 = node_1_xy
    x2, y2 = node_2_xy
    theta_1 = atan2(y1 - central_y, x1 - central_x)
    theta_2 = atan2(y2 - central_y, x2 - central_x)
    return atan2(sin(theta_1 - theta_2), cos(theta_1 - theta_2))

def kendalltau_comparison_v2(probe_graph,
                             true_positions,
                             recovered_positions,
                             central_node,
                             radius,
                             absolute=False,
                            ):
    subgraph = networkx.ego_graph(G=probe_graph,
                                  n=central_node,
                                  radius=radius,
                                  center=False,
                                 )
    probe_distances = {node: euclidean(true_positions[node],
                                       true_positions[central_node],
                                      )
                       for node in subgraph.nodes()
                       if node != central_node}
    #use nearest probe for theta
    nearest_node = min(probe_distances, key=probe_distances.get)
    if nearest_node == central_node:
        raise RuntimeError("Nearest node is central node.")
    elif probe_distances[nearest_node] == 0:
        raise RuntimeError("Nearest node on top of central node.")
    true_angles = []
    recovered_angles = []
    for node in subgraph:
        true_angle = node_angle_v2(central_node_xy=true_positions[central_node],
                                   node_1_xy=true_positions[nearest_node],
                                   node_2_xy=true_positions[node],
                                  )
        true_angles.append(true_angle)
        recovered_angle = node_angle_v2(central_node_xy=recovered_positions[central_node],
                                        node_1_xy=recovered_positions[nearest_node],
                                        node_2_xy=recovered_positions[node],
                                       )
        recovered_angles.append(recovered_angle)
    distance = kendalltau(true_angles, recovered_angles).correlation
    if absolute:
        distance = abs(distance)
    return distance

def unique_protein_in_tally(tally,
                            min_top_protein=10,
                            max_fraction_others=0.10,
                           ):
    total_counts = sum(tally.values())
    sorted_tally = sorted(tally.items(), key=lambda x:x[1], reverse=True)
    top_protein, top_count = sorted_tally[0]
    other_counts = sum([count for protein, count in sorted_tally[1:]])
    fraction_others = float(other_counts) / total_counts
    if top_count >= min_top_protein and fraction_others <= max_fraction_others:
        return True, top_protein
    else:
        return False, top_protein

def identify_unique_proteins(combined_observation_tallies,
                             min_top_protein,
                             max_fraction_others,
                            ):
    unique_proteins = set()
    for observation, tally in combined_observation_tallies.iteritems():
        assert len(tally) > 0, (s, observation)
        is_unique, top_protein = unique_protein_in_tally(tally=tally,
                                                         min_top_protein=min_top_protein,
                                                         max_fraction_others=max_fraction_others,
                                                        )
        if is_unique:
            unique_proteins.add(top_protein)
    return unique_proteins


def get_peripheral_nodes(probe_graph,
                         central_node,
                         radius,
                        ):
    outer_subgraph = networkx.ego_graph(G=probe_graph,
                                        n=central_node,
                                        radius=radius,
                                        center=True,
                                       )
    inner_subgraph = networkx.ego_graph(G=outer_subgraph,
                                        n=central_node,
                                        radius=radius - 1,
                                        center=True,
                                       )
    peripheral_nodes = set(outer_subgraph.nodes()) - set(inner_subgraph.nodes())
    return peripheral_nodes

def full_distortion_charachterization(probe_graph,
                                      central_node,
                                      radius,
                                      original_positions,
                                      distorted_positions,
                                     ):
    peripheral_nodes = get_peripheral_nodes(probe_graph=probe_graph,
                                            central_node=central_node,
                                            radius=radius,
                                           )
    center_to_periphery = {peripheral_node: (euclidean(original_positions[peripheral_node],
                                                       original_positions[central_node],
                                                      ),
                                             euclidean(distorted_positions[peripheral_node],
                                                       distorted_positions[central_node],
                                                      ),
                                            )
                           for peripheral_node in iter(peripheral_nodes)}


    original_hull = geometry.MultiPoint([original_positions[node]
                                         for node in iter(peripheral_nodes)]).convex_hull
    distorted_hull = geometry.MultiPoint([distorted_positions[node]
                                          for node in iter(peripheral_nodes)]).convex_hull
    areas = (original_hull.area, distorted_hull.area)

    nkotb = None

    return center_to_periphery, areas, nkotb


def get_best_alignment_with_flipping(probe_graph,
                                     original_positions,
                                     distorted_positions,
                                     comparison_radius,
                                     rotation_resolution=10,
                                    ):
    (nonflipped_best_alignment,
     nonflipped_best_distance,
     nonflipped_anchor_node,
    ) = get_best_alignment(probe_graph=probe_graph,
                           true_positions=original_positions,
                           recovered_positions=distorted_positions,
                           radius=comparison_radius,
                           flip_x=False,
                           flip_y=False,
                           resolution=rotation_resolution,
                          )
    rotated_distorted = rotate_graph(positions=distorted_positions,
                                     anchor_node=nonflipped_anchor_node,
                                     angle=nonflipped_best_alignment,
                                     flip_x=False,
                                     flip_y=False,
                                    )
    compare_distorted = anchor_translate(positions=rotated_distorted,
                                         anchor_node=nonflipped_anchor_node,
                                        )
    compare_original = anchor_translate(positions=original_positions,
                                        anchor_node=nonflipped_anchor_node,
                                       )
    nonflipped_values = {node: euclidean(compare_distorted[node], compare_original[node])
                         for node in probe_graph}



    (flipped_best_alignment,
     flipped_best_distance,
     flipped_anchor_node,
    ) = get_best_alignment(probe_graph=probe_graph,
                           true_positions=original_positions,
                           recovered_positions=distorted_positions,
                           radius=comparison_radius,
                           flip_x=True,
                           flip_y=False,
                           resolution=rotation_resolution,
                          )

    rotated_distorted = rotate_graph(positions=distorted_positions,
                                     anchor_node=flipped_anchor_node,
                                     angle=flipped_best_alignment,
                                     flip_x=True,
                                     flip_y=False,
                                    )
    compare_distorted = anchor_translate(positions=rotated_distorted,
                                         anchor_node=flipped_anchor_node,
                                        )
    compare_original = anchor_translate(positions=original_positions,
                                        anchor_node=flipped_anchor_node,
                                       )
    flipped_values = {node: euclidean(compare_distorted[node], compare_original[node])
                      for node in probe_graph}


    if sum(flipped_values.values()) >= sum(nonflipped_values.values()):
        flip_x = False
        return_values = nonflipped_values
    else:
        flip_x = True
        return_values = flipped_values

    return return_values, flip_x


def distortion_helper(probe_graph,
                      node,
                      rescaled_original_positions,
                      rescaled_refed_positions,
                      comparison_radius,
                      rotation_resolution,
                     ):
    center_to_periphery, areas, nkotb = \
        full_distortion_charachterization(probe_graph=probe_graph,
                                          central_node=node,
                                          radius=comparison_radius,
                                          original_positions=rescaled_original_positions,
                                          distorted_positions=rescaled_refed_positions,
                                         )

    neighborhood_subgraph = networkx.ego_graph(G=probe_graph,
                                               n=node,
                                               radius=comparison_radius,
                                               center=True,
                                              )
    neighborhood_rescaled_original_positions = \
                              {node: position
                               for node, position in rescaled_original_positions.iteritems()
                               if node in neighborhood_subgraph
                              }
    neighborhood_rescaled_refed_positions = {node: position
                                             for node, position in rescaled_refed_positions.iteritems()
                                             if node in neighborhood_subgraph
                                            }
    aligned_distances, flip_x = \
        get_best_alignment_with_flipping(probe_graph=neighborhood_subgraph,
                                         original_positions=neighborhood_rescaled_original_positions,
                                         distorted_positions=neighborhood_rescaled_refed_positions,
                                         comparison_radius=comparison_radius,
                                         rotation_resolution=rotation_resolution,
                                        )
    average_shift = np.mean(aligned_distances.values())
    median_shift = np.median(aligned_distances.values())
    std_shift = np.std(aligned_distances.values())

    distortion = (center_to_periphery,
                  areas,
                  nkotb,
                  average_shift,
                  median_shift,
                  std_shift,
                  aligned_distances,
                  flip_x,
                 )
    return distortion

def distortion_helper_MP(probe_graph,
                         rescaled_original_positions,
                         rescaled_refed_positions,
                         comparison_radius,
                         sample_size,
                         rotation_resolution,
                         num_processes=None,
                        ):
    if num_processes is None:
        num_processes = cpu_count()
    pool = Pool(processes=num_processes,
                maxtasksperchild=None,
               )
    processes = []
    per_process_nodes = [[] for p in range(num_processes)]
    distributor = cycle(per_process_nodes)
    for node in sample(probe_graph, sample_size):
        next_process = distributor.next()
        next_process.append(node)
    for nodes in per_process_nodes:
        for node in nodes:
            process = pool.apply_async(distortion_helper,
                                       (probe_graph,
                                        node,
                                        rescaled_original_positions,
                                        rescaled_refed_positions,
                                        comparison_radius,
                                        rotation_resolution,
                                       )
                                      )
            processes.append((node, process))
    pool.close()
    pool.join()
    distortions = {}
    for node, process in processes:
        distortion = process.get()
        distortions[node] = distortion
    return distortions
