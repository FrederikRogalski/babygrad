from graphviz import Digraph
from babygrad.value import Value

def graph(net, globs = None):
    d = Digraph()
    label = lambda x, var_name: ('\n'.join(var_name) + ("\n" if var_name else "") + 
                                 f"{x.symbol}\n" + 
                                 (f"Value: {x.value:.2f}" if x.value != None else "") + 
                                 f"\nGrad: {x.grad:.2f}")
    seen = set()
    globs = globs if globs else []
    def rec_graph(net, d: Digraph):
        # check if node is already in graph
        if id(net) in seen:
            return
        seen.add(id(net))
        
        var_name = [g for g in globs if globs[g]==net]
        d.node(str(id(net)), label(net, var_name), shape="rectangle" if isinstance(net.operands[0], Value) else "ellipse")
        
        for op in net.operands:
            if not isinstance(op, Value):
                return
            d.edge(str(id(net)), str(id(op)))
            rec_graph(op, d)
    rec_graph(net, d)
    return d