from pyvis.network import Network

def render_graph(nx_graph, height="500px"):
    net = Network(height=height, width="100%", directed=False)
    net.from_nx(nx_graph)
    return net.generate_html()