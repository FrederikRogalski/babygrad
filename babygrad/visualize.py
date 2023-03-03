import inspect
from graphviz import Digraph
from babygrad.value import Value, Operand, Operator

def graph(net, globs = None):
    d = Digraph()
    seen = set()
    globs = globs if globs else inspect.currentframe().f_back.f_locals
    def dfs(net: Operand, d: Digraph):
        if id(net) in seen:
            return
        seen.add(id(net))
        
        var_names = [g for g in globs if globs[g]==net]
        label = ",".join(var_names)+"\n" if var_names else ""
        # add symbol to label
        label += f"{net.symbol}\n"
        # add value
        label += f"{net.data}\n"
        # add gradient if attr grad exists
        label += f"grad: {net.grad}" if hasattr(net, "grad") else ""
        d.node(str(id(net)), label, shape="ellipse" if isinstance(net, Value) else "rectangle")
        if isinstance(net, Value):
            return
        net: Operator
        
        for operand in net.operands:
            d.edge(str(id(net)), str(id(operand)))
            dfs(operand, d)
    dfs(net, d)
    return d
        