import graphviz

dot = graphviz.Digraph(comment='System Architecture', format='png')
dot.attr(rankdir='TB', size='8,8')

# Nodes
dot.node('A', 'Orchestration Layer\n(MPI Manager)', shape='box', style='filled', fillcolor='lightblue')
dot.node('B', 'Acceleration Layer\n(CUDA/cuQuantum)', shape='box', style='filled', fillcolor='lightyellow')
dot.node('C', 'Quantum Interface\n(Dispatcher)', shape='box', style='filled', fillcolor='lightgrey')
dot.node('D', 'Cloud QPU', shape='cloud', style='filled', fillcolor='lightgreen')

#  Edges
dot.edge('A', 'B', label='Parameters θ')
dot.edge('B', 'C', label='OpenQASM')
dot.edge('C', 'D', label='Job Submission', color='red', style='dashed')
dot.edge('D', 'A', label='Expectation Values <H>')

dot.render('architecture_diagram')