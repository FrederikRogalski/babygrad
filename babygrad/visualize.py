import inspect
from graphviz import Digraph
from babygrad.tensor import LazyTensor, Operand, Operator

def _dfs(operand: Operand, visited: set[Operand], seq: list[Operand]):
    if operand in visited:
        return
    visited.add(operand)
    if isinstance(operand, Operator):
        for op in operand.operands:
            _dfs(op, visited, seq)
    seq.append(operand)

def graph(operand: Operand, rankdir="RL", bgcolor="#273348", fontcolor="#BBBBBB", color="#1A2332", requires_grad_color="#c2412f"):
    # digraph left to right
    d = Digraph()
    d.attr(rankdir=rankdir)
    d.attr(bgcolor=bgcolor)
    variables = inspect.currentframe().f_back.f_locals
    if isinstance(operand, LazyTensor):
        seq = [operand]
    elif isinstance(operand, Operator):
        _dfs(operand, set(), seq := [])
        seq.reverse()
    
    operand_to_name = {}
    for v in variables:
        try:
            if variables[v] in seq:
                operand_to_name[variables[v]] = v
        except ValueError:
            pass
    
    for operand in reversed(seq):
        children = [] if isinstance(operand, LazyTensor) else operand.operands
        _form = "oval" if isinstance(operand, LazyTensor) else "box"
        _color = requires_grad_color if operand.requires_grad or operand._decendant_requires_grad else color
        _fillcolor = requires_grad_color if operand.requires_grad else color
        _fontcolor = fontcolor
        d.node(str(id(operand)), _label(operand, operand_to_name), shape=_form, color=_color, fontcolor=_fontcolor, style="filled", fillcolor=_fillcolor, fontname="Menlo")
        d.edges([(str(id(operand)), str(id(child))) for child in children])
            
    return d

def _label(operand: Operand, operand_to_name: dict[Operand, str]):
    label = ""
    if operand in operand_to_name:
        label += f"{operand_to_name[operand]}\n"
    label += f"{operand.symbol}\n"
    trunc = 10
    # truncate data after 4 characters don't use :.4f because we don't know the type.
    label += f"{str(operand.data if operand.shape == () else operand.data.shape)[:trunc]}\n"
    label += f"\nGradient:\n{str(operand.grad if operand.grad is None else operand.grad.shape)[:trunc]}"
    return label
        