from __future__ import print_function
## Code used for plots (Taken from NEAT, few changes were made)

import copy
import warnings

import graphviz
import matplotlib.pyplot as plt
import numpy as np


def plot_stats(statistics, ylog=False, view=False, cnt = 0):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())
    best_std = np.array(np.std(best_fitness))

    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.fill_between(generation, avg_fitness - stdev_fitness, avg_fitness + stdev_fitness, alpha=0.2)
    plt.plot(generation, best_fitness, 'r-', label="best")
    plt.fill_between(generation, best_fitness - best_std, best_fitness + best_std, alpha=0.2)

    plt.title("Population's mean and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')
    filename = 'avg_fitness_' + str(cnt) + '.svg'
    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()

def plot_stats_nobest(statistics, ylog=False, view=False, cnt = 0):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.fill_between(generation, avg_fitness - stdev_fitness, avg_fitness + stdev_fitness, alpha=0.2)

    plt.title("Population's mean fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')
    filename = 'avg_fitness_nobest_' + str(cnt) + '.svg'
    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def plot_stats2(statistics, ylog=False, view=False, cnt = 0):
    """ Plots the population's median and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    median_fitness = np.array(statistics.get_fitness_median())
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    stat25 = []
    stat75 = []
    for st in statistics.generation_statistics:
        scores = []
        for species_stats in st.values():
            scores.extend(species_stats.values())
        stat25.append(np.percentile(scores, 25))
        stat75.append(np.percentile(scores, 75))

    best25 = np.percentile(np.array(best_fitness),25)
    best75 = np.percentile(np.array(best_fitness), 75)

    plt.plot(generation, median_fitness, 'b-', label="median")
    plt.fill_between(generation, stat25, stat75, alpha=0.2)
    plt.plot(generation, best_fitness, 'r-', label="best")
    plt.fill_between(generation, best25, best75, alpha=0.2)

    plt.title("Population's best, median, 25th and 75th quartile")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')
    filename = 'median_fitness_' + str(cnt) + '.svg'
    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()

def plot_stats2_nobest(statistics, ylog=False, view=False, cnt = 0):
    """ Plots the population's median and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    median_fitness = np.array(statistics.get_fitness_median())
    stat25 = []
    stat75 = []
    for st in statistics.generation_statistics:
        scores = []
        for species_stats in st.values():
            scores.extend(species_stats.values())
        stat25.append(np.percentile(scores, 25))
        stat75.append(np.percentile(scores, 75))

    plt.plot(generation, median_fitness, 'b-', label="median")
    plt.fill_between(generation, stat25, stat75, alpha=0.2)

    plt.title("Population's median, 25th and 75th quartile")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')
    filename = 'median_fitness_nobest_' + str(cnt) + '.svg'
    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()

def plot_head_balance(statistics, head_balance,  ylog=False, view=False, cnt = 0):
    """ Plots the avg. and best head balance of the robots across generations. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    mean_lst = []
    max_lst = []
    std_lst = []
    for lst in head_balance:
        mean_lst.append(np.mean(lst))
        max_lst.append(np.max(lst))
        std_lst.append(np.std(lst))

    mean_lst = np.array(mean_lst)
    std_lst = np.array(std_lst)
    max_lst = np.array(max_lst)
    max_std = np.std(max_lst)

    plt.plot(generation, mean_lst, 'b-', label="avg. head balance ")
    plt.fill_between(generation, mean_lst - std_lst, mean_lst + std_lst, alpha=0.2)
    plt.plot(generation, max_lst, 'r-', label="best head balance")
    plt.fill_between(generation, max_lst - max_std, max_lst + max_std, alpha=0.2)

    plt.title("Head Balance across generations (1 being the most balanced)")
    plt.xlabel("Generations")
    plt.ylabel("Balance")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')
    filename = 'head_balance_' + str(cnt) + '.svg'
    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()

def plot_head_balance_nobest(statistics, head_balance,  ylog=False, view=False, cnt = 0):
    """ Plots the avg. and best head balance of the robots across generations. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    mean_lst = []
    std_lst = []
    for lst in head_balance:
        mean_lst.append(np.mean(lst))
        std_lst.append(np.std(lst))

    mean_lst = np.array(mean_lst)
    std_lst = np.array(std_lst)

    plt.plot(generation, mean_lst, 'b-', label="avg. head balance ")
    plt.fill_between(generation, mean_lst - std_lst, mean_lst + std_lst, alpha=0.2)

    plt.title("Head Balance across generations (1 being the most balanced)")
    plt.xlabel("Generations")
    plt.ylabel("Balance")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')
    filename = 'head_balance_nobest_' + str(cnt) + '.svg'
    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()

def plot_stats_avg(best, avg, stdev, gens, ylog=False, view=False, filename='avg_fitness.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(gens)
    best_fitness = best
    avg_fitness = np.array(avg)
    stdev_fitness = np.array(stdev)
    best_std = np.std(np.array(best_fitness))

    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.fill_between(generation, avg_fitness - stdev_fitness, avg_fitness + stdev_fitness, alpha=0.2)
    plt.plot(generation, best_fitness, 'r-', label="best")
    plt.fill_between(generation, best_fitness - best_std, best_fitness + best_std, alpha=0.2)

    plt.title("Population's mean, stdev and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def plot_stats_avg_nobest(best, avg, stdev, gens, ylog=False, view=False, filename='avg_fitness_nobest.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(gens)
    avg_fitness = np.array(avg)
    stdev_fitness = np.array(stdev)

    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.fill_between(generation, avg_fitness - stdev_fitness, avg_fitness + stdev_fitness, alpha=0.2)

    plt.title("Population's mean and stdev fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()

def plot_stats2_avg(median, stat25, stat75, best, gens, ylog=False, view=False, filename='median_fitness.svg'):
    """ Plots the population's median and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(gens)
    median_fitness = np.array(median)
    best_fitness = best
    best25 = np.percentile(np.array(best_fitness), 25)
    best75 = np.percentile(np.array(best_fitness), 75)

    plt.plot(generation, median_fitness, 'b-', label="median")
    plt.fill_between(generation, stat25, stat75, alpha=0.2)
    plt.plot(generation, best_fitness, 'r-', label="best")
    plt.fill_between(generation, best25, best75, alpha=0.2)

    plt.title("Population's best, median, 25th and 75th quartile")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def plot_stats2_avg_nobest(median, stat25, stat75, best, gens, ylog=False, view=False, filename='median_fitness_nobest.svg'):
    """ Plots the population's median and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(gens)
    median_fitness = np.array(median)

    plt.plot(generation, median_fitness, 'b-', label="median")
    plt.fill_between(generation, stat25, stat75, alpha=0.2)

    plt.title("Population's median, 25th and 75th quartile")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()

def plot_head_balance_avg(mean, max, std, gens,  ylog=False, view=False, filename='head_balance.svg'):
    """ Plots the avg. and best head balance of the robots across generations. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(gens)

    mean_lst = np.array(mean)
    std_lst = np.array(std)
    max_lst = np.array(max)
    max_std = np.std(max_lst)

    plt.plot(generation, mean_lst, 'b-', label="avg. head balance ")
    plt.fill_between(generation, mean_lst - std_lst, mean_lst + std_lst, alpha=0.2)
    plt.plot(generation, max_lst, 'r-', label="best head balance")
    plt.fill_between(generation, max_lst - max_std, max_lst + max_std, alpha=0.2)

    plt.title("Head Balance across generations (1 being the most balanced)")
    plt.xlabel("Generations")
    plt.ylabel("Balance")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def plot_head_balance_avg_nobest(mean, max, std, gens, ylog=False, view=False, filename='head_balance_nobest.svg'):
    """ Plots the avg. and best head balance of the robots across generations. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(gens)

    mean_lst = np.array(mean)
    std_lst = np.array(std)

    plt.plot(generation, mean_lst, 'b-', label="avg. head balance ")
    plt.fill_between(generation, mean_lst - std_lst, mean_lst + std_lst, alpha=0.2)

    plt.title("Head Balance across generations (1 being the most balanced)")
    plt.xlabel("Generations")
    plt.ylabel("Balance")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()

def plot_spikes(spikes, view=False, filename=None, title=None):
    """ Plots the trains for a single spiking neuron. """
    t_values = [t for t, I, v, u, f in spikes]
    v_values = [v for t, I, v, u, f in spikes]
    u_values = [u for t, I, v, u, f in spikes]
    I_values = [I for t, I, v, u, f in spikes]
    f_values = [f for t, I, v, u, f in spikes]

    fig = plt.figure()
    plt.subplot(4, 1, 1)
    plt.ylabel("Potential (mv)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, v_values, "g-")

    if title is None:
        plt.title("Izhikevich's spiking neuron model")
    else:
        plt.title("Izhikevich's spiking neuron model ({0!s})".format(title))

    plt.subplot(4, 1, 2)
    plt.ylabel("Fired")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, f_values, "r-")

    plt.subplot(4, 1, 3)
    plt.ylabel("Recovery (u)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, u_values, "r-")

    plt.subplot(4, 1, 4)
    plt.ylabel("Current (I)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, I_values, "r-o")

    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()
        plt.close()
        fig = None

    return fig


def plot_species(statistics, view=False, cnt = 0):
    """ Visualizes speciation throughout evolution. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")
    filename = 'speciation_' + str(cnt) + '.svg'
    plt.savefig(filename)

    if view:
        plt.show()

    plt.close()


def draw_net(config, genome, view=False, cnt = 0, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled',
                       'shape': 'box'}
        input_attrs['fillcolor'] = node_colors.get(k, 'lightgray')
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled'}
        node_attrs['fillcolor'] = node_colors.get(k, 'lightblue')

        dot.node(name, _attributes=node_attrs)

    if prune_unused:
        connections = set()
        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                connections.add((cg.in_node_id, cg.out_node_id))

        used_nodes = copy.copy(outputs)
        pending = copy.copy(outputs)
        while pending:
            new_pending = set()
            for a, b in connections:
                if b in pending and a not in used_nodes:
                    new_pending.add(a)
                    used_nodes.add(a)
            pending = new_pending
    else:
        used_nodes = set(genome.nodes.keys())

    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled',
                 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            #if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    filename = 'digraph_' + str(cnt) + '.gv'
    dot.render(filename, view=view)

    return dot