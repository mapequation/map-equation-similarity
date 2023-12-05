import matplotlib.pyplot as plt
import networkx          as nx
import numpy             as np
import seaborn           as sb

from .mapsim import MapSim
from .util   import address_path, bspline, inits

def plot_hierarchy(mapsim : MapSim, G : nx.Graph) -> None:
    # node addresses to flow
    node_to_flow = { address : mapsim.cb.get_flow(address) for address in mapsim.addresses.values() }

    # a reverse mapping from addresses to nodes
    address_to_node = { v:k for k,v in mapsim.addresses.items()}

    # the nodes that represent modules, as opposed to actual nodes
    module_nodes = set()

    for address in mapsim.addresses.values():
        for init in inits(address):
            module_nodes.add(init)
    
    # the flows for the module nodes
    module_node_to_flow = dict()
    for module_node in module_nodes:
        sub_codebook = mapsim.cb.get_module(module_node)
        module_node_to_flow[module_node] = sub_codebook.flow
    
    # the actual plotting...
    fig, ax = plt.subplots(1, 1, figsize = (6,6))

    palette = sb.color_palette("colorblind")

    # radial positions for all nodes
    radial_pos = dict()
    radial_pos[()] = (0,0)

    # calculate node positions on the disc
    def child_poincare(x,y,r,theta):
        x_ = x + r * np.cos(theta)
        y_ = y + r * np.sin(theta)

        return (x_,y_)

    # the nodes' modules
    modules = dict()

    theta = 0
    for (address, flow) in node_to_flow.items():
        node = address_to_node[address]

        # super-crude way to decide what module the node belongs to
        modules[node] = address[0]
        module = address[0]

        theta += flow * np.pi
        p = child_poincare(0, 0, r = 2, theta = theta)
        radial_pos[address] = p
        theta += flow * np.pi
        ax.pie( [flow] # flows
            , colors = [palette[(module-1) % len(palette)]]
            , center = p
            , radius = 0.5 * np.sqrt(flow)
            , startangle = 0 # startangle
            , wedgeprops = { "linewidth": 1, "edgecolor": "white" }
            )

    plt.scatter([0], [0], marker = "s", c = ["grey"])

    angle_offsets = {():0}
    for address in sorted(module_nodes, key = lambda addr: (len(addr), addr)):
        # get angle offset for *this* node
        theta = angle_offsets[address[:-1]]

        # and remember the offset for potential children
        angle_offsets[address] = theta

        theta += module_node_to_flow[address] * np.pi
        r = sum([1/2**i for i in range(len(address))])
        p = child_poincare(0, 0, r = r, theta = theta)
        radial_pos[address] = p
        theta += module_node_to_flow[address] * np.pi

        # and update the angle offset for siblings
        angle_offsets[address[:-1]] = theta

        parent = radial_pos[address[:-1]]

        plt.plot([parent[0],p[0]], [parent[1],p[1]], c = "grey", alpha = 0.5)
        plt.scatter([p[0]], [p[1]], marker = "s", c = [palette[(address[0] - 1) % len(palette)]])

    for (u, v) in G.edges:
        source = mapsim.addresses[u]
        target = mapsim.addresses[v]
        path = address_path(source = list(source), target = list(target))
        points = np.array([radial_pos[tuple(address)] for address in path])
        #bps = np.array([BezierSegment(points).point_at_t(x) for x in np.linspace(0,1,num = 100)])
        bps = bspline(points, n = 100, degree = len(path)-1)

        if modules[u] == modules[v]:
            colour = palette[(modules[u]-1) % len(palette)]
            plt.plot(bps[:,0], bps[:,1], color = colour, alpha = 0.8)
        else:
            colour_u = palette[(modules[u]-1) % len(palette)]
            colour_v = palette[(modules[v]-1) % len(palette)]

            for (ix, (p,q)) in enumerate(zip(bps, bps[1:])):
                frac = ix / len(bps)
                colour = (1-frac) * np.array(colour_u) + frac * np.array(colour_v)
                plt.plot([p[0], q[0]], [p[1], q[1]], color = colour, alpha = 0.8)

    ax.axis("off")
    plt.autoscale()
    plt.show()