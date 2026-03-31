import networkx as nx
import heapq


# A utility function to find root or origin of the node i in MST
def findRoot(parent, i):
    if parent[i] == i:
        return i
    return findRoot(parent, parent[i])

# A function that does union of set x and y based on the order
def union(parent, order, x, y):
    xRoot = findRoot(parent, x)
    yRoot = findRoot(parent, y)
 	# Attach smaller order tree under root of high order tree
    if order[xRoot] < order[yRoot]:
        parent[xRoot] = yRoot
    elif order[xRoot] > order[yRoot]:
        parent[yRoot] = xRoot
    # If orders are same, then make any one as root and increment its order by one
    else :
        parent[yRoot] = xRoot
        order[xRoot] += 1

# takes input from the file and creates a weighted graph
def CreateGraph(matrix, thr=0):
  G = nx.Graph()
  wtMatrix = matrix
  n = len(wtMatrix)

	# Adds egdes along with their weights to the graph 
  for i in range(n) :
    for j in range(n)[i:] :
      # if wtMatrix[i][j] > thr :
      G.add_edge(i, j, length = wtMatrix[i][j]) 
  return G

# draws the graph and displays the weights on the edges
def DrawGraph(G, plot=False):
	pos = nx.spring_layout(G)
	if plot: nx.draw(G, pos, with_labels = True)  # with_labels=true is to show the node number in the output graph
	edge_labels = nx.get_edge_attributes(G, 'length')
	if plot: nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels, font_size = 11) #    prints weight on all the edges
	return pos

def kruskal_optimized(G, image_list=None, ransac_inliers=None, thr=30, inlier_key='ransac_inliers'):
    edges = [(edata['length'], u, v) for u, v, edata in G.edges(data=True) if 'length' in edata]
    heapq.heapify(edges)

    node_list = list(G.nodes())
    node_index = {node: i for i, node in enumerate(node_list)}
    vLen = len(node_list)
    parent = list(range(vLen))
    rank = [0] * vLen
    mst = []

    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(x, y):
        xroot = find(x)
        yroot = find(y)
        if xroot != yroot:
            if rank[xroot] < rank[yroot]:
                parent[xroot] = yroot
            elif rank[xroot] > rank[yroot]:
                parent[yroot] = xroot
            else:
                parent[yroot] = xroot
                rank[xroot] += 1
    reject_times = 0
    ransac_times = 0
    reject_edges = []

    while len(mst) < vLen - 1 and edges:
        length, u, v = heapq.heappop(edges)
        u_idx = node_index[u]
        v_idx = node_index[v]
        if find(u_idx) != find(v_idx):
            if ransac_inliers is not None:
              num_inliers = ransac_inliers[f"{image_list[u_idx]}_{image_list[v_idx]}"]
              ransac_times += 1
              if inlier_key != 'R':
                if num_inliers < thr: 
                  reject_times += 1
                  reject_edges += [(u, v)]
                  continue
              else:
                if num_inliers > thr: 
                  reject_times += 1
                  reject_edges += [(u, v)]
                  continue
              
            mst.append((u, v, length))
            union(u_idx, v_idx)

    if len(reject_edges) == 0: 
       return mst
    else: 
       return mst, ransac_times, reject_times, reject_edges


