import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the adjacency matrix we saved in Step 1
adj = pd.read_csv("data/graph_edges.csv", index_col=0)

# Visualize the 'Edges' of Tyche's world
plt.figure(figsize=(8, 6))
sns.heatmap(adj, annot=True, cmap='RdYlGn')
plt.title("Tyche Pillar 4: Market Graph Edges (Correlations)")
plt.show()