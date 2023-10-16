import itertools
import matplotlib.pyplot as plt
import networkx          as nx
import numpy             as np
import seaborn           as sb
import scipy

from collections     import Counter, defaultdict
from progress.bar    import Bar
from infomap         import Infomap
from sklearn.metrics import adjusted_mutual_info_score
from mapsim          import MapSim
from typing          import List

fst = lambda p: p[0]
snd = lambda p: p[1]

def mkProbabilities(G, mapsim, beta = 1, use_source_degree = False, use_target_degree = False):
    similarities  = defaultdict(lambda: defaultdict(float))
    likelihoods   = defaultdict(lambda: defaultdict(float))
    probabilities = defaultdict(lambda: defaultdict(float))
    
    population = lambda u: G.degree(u) if use_source_degree else 1
    popularity = lambda v: G.degree(v) if use_target_degree else 1
    #population = lambda u: mapsim.cb.get_flow(mapsim.addresses[u])
    
    for u in G.nodes:
        for v in G.nodes:
            if v != u:
                similarities[u][v] = mapsim.get_path_cost_directed(u,v)
    
    for u in G.nodes:
        for v in G.nodes:
            if v != u:
                likelihoods[u][v] = popularity(v) * 2**(-beta * similarities[u][v])
    
    for u in G.nodes:
        likelihood_sum = sum(likelihoods[u].values())
        for v in G.nodes:
            if v != u:
                probabilities[u][v] = population(u) * likelihoods[u][v] / likelihood_sum

    return probabilities


# def mkProbabilities(G, mapsim, beta = 1, use_source_degree = False, use_target_degree = False):
#     similarities  = { u : mapsim.predict_interaction_rates(u, include_self_links = False) for u in G.nodes }
#     likelihoods   = defaultdict(lambda: defaultdict(float))
#     probabilities = defaultdict(lambda: defaultdict(float))
    
#     population = lambda u: G.degree(u) if use_source_degree else 1
#     popularity = lambda v: G.degree(v) if use_target_degree else 1

#     #population = lambda u: mapsim.cb.get_flow(mapsim.addresses[u])
    
#     for u in G.nodes:
#         for v in G.nodes:
#             if v != u:
#                 likelihoods[u][v] = popularity(v) * np.exp(beta * similarities[u][v])
    
#     for u in G.nodes:
#         likelihood_sum = sum(likelihoods[u].values())
#         for v in G.nodes:
#             if v != u:
#                 probabilities[u][v] = population(u) * likelihoods[u][v] / likelihood_sum

#     return probabilities


def mkStatistics(G : nx.Graph, mapsim : MapSim, betas : List[float], k : int = 10):
    if G.is_directed():
        out_degree_distributions = defaultdict(lambda: list())
        in_degree_distributions  = defaultdict(lambda: list())
    else:
        degree_distributions = defaultdict(lambda: list())
    community_assignments    = defaultdict(lambda: list())
    number_of_edges          = defaultdict(lambda: list())
    mixing                   = defaultdict(lambda: list())

    for beta in Bar("Betas", check_tty = False).iter(betas):
        probs = mkProbabilities(G = G, mapsim = mapsim, beta = beta, use_source_degree = True, use_target_degree = False)
        
        for _ in range(k):
            G_sampled = nx.DiGraph() if G.is_directed() else nx.Graph()
            G_sampled.add_nodes_from(G.nodes)

            for u,ps in probs.items():
                rs = np.random.rand(G.number_of_nodes())
                for (v,p),r in zip(ps.items(), rs):
                    if r <= p:
                        G_sampled.add_edge(u,v)

            infomap_sampled = Infomap(silent = True, num_trials = 100, two_level = True)
            node_mapping = infomap_sampled.add_networkx_graph(G_sampled)
            infomap_sampled.run()
            
            if G.is_directed():
                out_degree_distributions[beta].append(Counter(dict(G_sampled.out_degree).values()))
                in_degree_distributions[beta].append(Counter(dict(G_sampled.in_degree).values()))
            else:
                degree_distributions[beta].append(Counter(dict(G_sampled.degree).values()))
            community_assignments[beta].append(list(dict(infomap_sampled.modules).values()))
            number_of_edges[beta].append(G_sampled.number_of_edges())
            
            modules = {node_mapping[k]:v for k,v in dict(infomap_sampled.modules).items()}

            within  = 0
            between = 0
            for (u,v) in G_sampled.edges:
                if modules[u] == modules[v]:
                    within += 1
                else:
                    between += 1
            mixing[beta].append(between / (within + between))

    
    res = dict( community_assignments = community_assignments
              , number_of_edges       = number_of_edges
              , mixing                = mixing
              )
    
    if G.is_directed():
        res["out_degree_distributions"] = out_degree_distributions
        res["in_degree_distributions"]  = in_degree_distributions
    else:
        res["degree_distributions"] = degree_distributions

    return res


def plotResults( G : nx.Graph
               , pos
               , infomap
               , node_mapping
               , mapsim
               , modules
               , betas
               , betas_ixs
               , network_name
               , k
               , avg_degree_distributions
               , community_assignments
               , number_of_edges
               , angled_links = False
               , maxx         = None
               , maxy         = None
               ):
    fig,axss = plt.subplots(2, len(betas_ixs) + 1, figsize = (3.6 * (len(betas_ixs) + 1), 6.5))
    axs      = list(itertools.chain.from_iterable(axss))
    palette  = sb.color_palette("viridis", infomap.num_leaf_modules)

    #pos = nx.kamada_kawai_layout(G)
    nx.draw_networkx_nodes(G, pos = pos, ax = axs[0], node_color = [palette[m-1] for m in modules.values()], node_size = 30)
    nx.draw_networkx_edges(G, pos = pos, ax = axs[0], edge_color = sb.color_palette()[3], connectionstyle = f"arc3,rad={0.2 if angled_links else 0}", arrows = True, node_size = 30)
    axs[0].set_title(f"{G.number_of_edges()} links, |M| = {infomap.num_leaf_modules}")
    axs[0].axis("off")

    original_communities = list(dict(infomap.modules).values())

    yss = [[adjusted_mutual_info_score(original_communities, dd) for dd in community_assignments[beta]] for beta in betas]

    # for beta,ys in zip(betas,yss):
    #     axs[len(betas_ixs) + 1].scatter(len(ys)*[beta], ys, alpha = 0.25, color = sb.color_palette()[0], marker = ".")

    errors = [np.mean(ys) - scipy.stats.t.interval(0.95, len(ys)-1, loc = np.mean(ys), scale = scipy.stats.sem(ys))[0] for ys in yss]

    axs[len(betas_ixs) + 1].scatter(betas, [np.average(ys) for ys in yss], color = sb.color_palette()[0], marker = "o", s = 10)
    axs[len(betas_ixs) + 1].errorbar(betas, [np.average(ys) for ys in yss], errors, color = sb.color_palette()[0], linestyle = "", alpha = 0.75)
    axs[len(betas_ixs) + 1].plot(betas, [np.average(ys) for ys in yss], linestyle = "", color = sb.color_palette()[0])
    axs[len(betas_ixs) + 1].set_xlabel("$\\beta$")
    axs[len(betas_ixs) + 1].set_ylabel("AMI", color = palette[0])
    axs[len(betas_ixs) + 1].set_xlim(0, max(betas))
    axs[len(betas_ixs) + 1].set_title(f"averages ({k} samples)")


    twin_ax = axs[len(betas_ixs) + 1].twinx()

    errors = [np.mean(number_of_edges[beta]) - scipy.stats.t.interval(0.95, len(number_of_edges[beta])-1, loc = np.mean(number_of_edges[beta]), scale = scipy.stats.sem(number_of_edges[beta]))[0] for beta in betas]

    twin_ax.scatter(betas, [np.average(number_of_edges[beta]) for beta in betas], color = sb.color_palette()[1], marker = "o", s = 10)
    twin_ax.errorbar(betas, [np.average(number_of_edges[beta]) for beta in betas], errors, color = sb.color_palette()[1], linestyle = "", alpha = 0.75)
    twin_ax.plot(betas, [np.average(number_of_edges[beta]) for beta in betas], color = sb.color_palette()[1], linestyle = "")
    twin_ax.hlines(G.number_of_edges(), 0, max(betas), color = sb.color_palette()[1], linestyles = "--")
    twin_ax.set_ylabel("#links", color = sb.color_palette()[1])

    xmaxs = []
    ymaxs = []

    #for axb, axa, sampled_beta in [(1,6,betas[0]), (2,7,betas[len(betas)//3]), (3,8,betas[2*len(betas)//3]), (4,9,betas[-1])]:
    #for axb, axa, sampled_beta in [(1,6,plot_betas[0]), (2,7,plot_betas[1]), (3,8,plot_betas[2]), (4,9,plot_betas[3])]:
    for ix, beta_ix in enumerate(betas_ixs, start = 1):
        sampled_beta = betas[beta_ix]
        axa          = axs[ix + len(betas_ixs) + 1]
        axb          = axs[ix]

        if G.is_directed():
            xs, ys = zip(*sorted(Counter(dict(G.out_degree).values()).items()))
            axa.bar(xs, ys, label = "original out", color = sb.color_palette()[0], alpha = 0.5)

            xs, ys = zip(*sorted(Counter(dict(G.in_degree).values()).items()))
            axa.bar(xs, ys, label = "original in", color = sb.color_palette()[3], alpha = 0.5)

            avg_out_degree_distributions, avg_in_degree_distributions = avg_degree_distributions
            
            xs, ys = zip(*sorted(avg_out_degree_distributions[sampled_beta].items()))
            axa.bar(xs, ys, label = "sampled out", color = sb.color_palette()[1], alpha = 0.5)

            xs, ys = zip(*sorted(avg_in_degree_distributions[sampled_beta].items()))
            axa.bar(xs, ys, label = "sampled in", color = sb.color_palette()[2], alpha = 0.5)
        
        else:
            xs, ys = zip(*sorted(Counter(dict(G.degree).values()).items()))
            axa.bar(xs, ys, label = "original", color = sb.color_palette()[0], alpha = 0.5)

            xs, ys = zip(*sorted(avg_degree_distributions[sampled_beta].items()))
            axa.bar(xs, ys, label = "sampled", color = sb.color_palette()[1], alpha = 0.5)
        
        axa.set_title(f"degree distribution ({k} samples)")
        axa.legend()
        xmaxs.append(axa.get_xlim()[1])
        ymaxs.append(axa.get_ylim()[1])
        
        probs = mkProbabilities(G, mapsim, beta = sampled_beta, use_source_degree = True, use_target_degree = False)
        
        G_sampled = nx.DiGraph() if G.is_directed() else nx.Graph()
        G_sampled.add_nodes_from(G.nodes)

        for u,ps in probs.items():
            for v,p in ps.items():
                if np.random.rand() <= p:
                    G_sampled.add_edge(u,v)

        infomap_sampled = Infomap(silent = True, num_trials = 100, two_level = True)
        infomap_sampled.add_networkx_graph(G_sampled)
        infomap_sampled.run()
        
        modules_sampled = {node_mapping[k]:v for k,v in dict(infomap_sampled.modules).items()}

        palette = sb.color_palette("viridis", infomap_sampled.num_leaf_modules)

        nx.draw_networkx_nodes( G_sampled
                              , pos = pos
                              , ax = axb
                              , node_color = [palette[m-1] for m in modules_sampled.values()]
                              , node_size = 30
                              )
        nx.draw_networkx_edges( G_sampled
                            , pos = pos
                            , ax = axb
                            , edge_color = sb.color_palette()[3]
                            , node_size = 30
                            , connectionstyle = f"arc3,rad={0.2 if angled_links else 0}"
                            , arrows = True
                            )
        axb.set_title(f"$\\beta$ = {sampled_beta:.1f}, {G_sampled.number_of_edges()} links, |M| = {len(set(modules_sampled.values()))}")
        axb.axis("off")

    for ax in range(len(betas_ixs) + 2, 2*len(betas_ixs)+2): # [6,7,8,9]:
        if maxx is not None:
            axs[ax].set_xlim(0, min(maxx,max(xmaxs)))
        else:
            axs[ax].set_xlim(0, max(xmaxs))

        if maxy is not None:
            axs[ax].set_ylim(0, min(maxy, max(ymaxs)))
        else:
            axs[ax].set_ylim(0, max(ymaxs))

        axs[ax].set_xlabel("degree")
        axs[ax].set_ylabel("count")

        
    fig.tight_layout()
    plt.savefig(f"{network_name}.pdf", bbox_inches = "tight", transparent = True)
    plt.show()