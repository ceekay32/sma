import pandas as pd
final_df=pd.read_csv("preprocessed_tweets_v2.csv")

cityClass=["Mumbai","Delhi","Bengaluru South","Lucknow","Pune"]
hashtagsClass=["LokSabhaElections2019","PMModi","Elections2019","Votekar"]
freq_of_tweets={"Mumbai":{"LokSabhaElections2019":0,"PMModi":0,"Elections2019":0,"Votekar":0},"Delhi":{"LokSabhaElections2019":0,"PMModi":0,"Elections2019":0, "Votekar":0},"Pune":{"LokSabhaElections2019":0,"PMModi":0,"Elections2019":0 ,"Votekar":0},"Lucknow":{"LokSabhaElections2019":0,"PMModi":0,"Elections2019":0, "Votekar":0}, "Bengaluru South":{"LokSabhaElections2019":0,"PMModi":0,"Elections2019":0,"Votekar":0}}

from collections import Counter
cnt = Counter()
for text in final_df["Place Name"].values:
    # for word in text.split():
        cnt[text] += 1
        
print(cnt.most_common(15))

cities=final_df["Place Name"]
hash=final_df['Hashtags']

for i in range(0, len(final_df)):
  for city in cityClass:
    if city in cities[i]:
      for tag in hashtagsClass:
        if tag in hash[i]:
          freq_of_tweets[city][tag] += 1
          break
        else:
          continue
    else:
      continue
print(freq_of_tweets)

#-------------------------Network Graph-------------------------
import networkx as nx
import matplotlib.pyplot as plt
G = nx.DiGraph()
edge_lab = {}
for city in freq_of_tweets:
  for tag in freq_of_tweets[city]:
    G.add_edge(city, tag, weight = freq_of_tweets[city][tag])
    edge_lab[city, tag] = freq_of_tweets[city][tag]

pos = nx.bipartite_layout(G,cityClass)
plt.figure(figsize =(10, 10))
nx.draw(G,pos, with_labels=True, node_size = 1500)
nx.draw_networkx_edge_labels(G, pos, edge_labels =edge_lab)
plt.show

#-------------------------Community detection using Spectral Clustering-------------------------
import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering

# Load your graph
G = nx.DiGraph()
for city in freq_of_tweets:
  for tag in freq_of_tweets[city]:
    G.add_edge(city, tag, weight=freq_of_tweets[city][tag])

# Convert the graph to an adjacency matrix
adj_matrix = nx.adjacency_matrix(G).toarray()

# Perform spectral clustering to detect communities
spectral = SpectralClustering(n_clusters=3, affinity='precomputed')
spectral.fit(adj_matrix)

# Print the communities
labels = spectral.labels_
unique_labels = np.unique(labels)
for i, com in enumerate(unique_labels):
  print("Community", i)
  nodes = [node for j, node in enumerate(G.nodes()) if labels[j] == com]
  print(nodes)

#-------------------------Plot the communities-------------------------
import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt



# Get the positions of the nodes for plotting
pos = nx.bipartite_layout(G, cityClass)

# Plot the graph with nodes colored by community
plt.figure(figsize=(10, 10))
nx.draw_networkx_nodes(G, pos, nodelist=[node for i, node in enumerate(G.nodes()) if spectral.labels_[i] == 0], node_color='b', node_size=1500)
nx.draw_networkx_nodes(G, pos, nodelist=[node for i, node in enumerate(G.nodes()) if spectral.labels_[i] == 1], node_color='r', node_size=1500)
nx.draw_networkx_nodes(G, pos, nodelist=[node for i, node in enumerate(G.nodes()) if spectral.labels_[i] == 2], node_color='g', node_size=1500)

nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
nx.draw_networkx_edge_labels(G, pos, edge_labels =edge_lab)

plt.show()

#--------------------------------------------------ALTERNATE--------------------------------------------------

temp_df=pd.read_csv("preprocessed_tweets_v2.csv")
print(temp_df["Source Label"].unique())


isVerified=["TRUE","FALSE"]
sourceLabel=['Twitter for Android', 'Twitter for iPhone', 'Twitter Web Client','Twitter for iPad', 'Tweetbot for Mac', 'UberSocial© PRO']
freq_of_tweets={"TRUE":{"Twitter for Android":0,"Twitter for iPhone":0,"Twitter Web Client":0, "Twitter for iPad":0,"Tweetbot for Mac":0,"UberSocial© PRO":0},"FALSE":{"Twitter for Android":0,"Twitter for iPhone":0,"Twitter Web Client":0, "Twitter for iPad":0,"Tweetbot for Mac":0,"UberSocial© PRO":0}}

isVerified=["TRUE","FALSE"]
sourceLabel=['Twitter for Android', 'Twitter for iPhone', 'Twitter Web Client','Twitter for iPad' ,'Hootsuite Inc.', 'Tweetbot for iΟS', 'erased972529_fzyRVGtcON', 'Tweetbot for Mac']
freq_of_tweets={"TRUE":{'Twitter for Android':0, 'Twitter for iPhone':0, 'Twitter Web Client':0,'Twitter for iPad':0,'Hootsuite Inc.':0, 'Tweetbot for iΟS':0,'erased972529_fzyRVGtcON' :0,'Tweetbot for Mac':0},"FALSE":{'Twitter for Android':0, 'Twitter for iPhone':0, 'Twitter Web Client':0,'Twitter for iPad':0,'Hootsuite Inc.':0, 'Tweetbot for iΟS':0,'erased972529_fzyRVGtcON' :0,'Tweetbot for Mac':0}}

verified=temp_df["isVerified"]
srcLabel=temp_df['Source Label']

for i in range(0, len(temp_df)):
    verify = str(verified[i]).upper()
    freq_of_tweets[verify][srcLabel[i]] += 1

print(freq_of_tweets)

#-------------------------NETWORK GRAPH-------------------------
import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()

for verify in isVerified:
    G.add_node(verify)
for tag in sourceLabel:
    G.add_node(tag)

for verify in freq_of_tweets:
    for tag in freq_of_tweets[verify]:
        G.add_edge(verify, tag, weight=freq_of_tweets[verify][tag])

plt.figure(figsize=(40, 40))
pos = nx.spring_layout(G, k=0.5, iterations=1)  # Adjust parameters here
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=10000)
nx.draw_networkx_edges(G, pos, width=2)
nx.draw_networkx_labels(G, pos, font_size=30)
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): w['weight'] for u, v, w in G.edges(data=True)}, font_size=30)
plt.show()

#-------------------------COMMUNTIY DETECTION-------------------------
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# create the network graph
G = nx.Graph()
for verify in freq_of_tweets:
    for tag in freq_of_tweets[verify]:
        G.add_edge(verify, tag, weight=freq_of_tweets[verify][tag])

# apply hierarchical clustering
model = AgglomerativeClustering(n_clusters=None, distance_threshold=0)
model.fit_predict(nx.to_numpy_array(G))

# draw the network graph with community labels
plt.figure(figsize=(40, 40))
pos = nx.spring_layout(G, k=0.5, iterations=1)  # Adjust parameters here
nx.draw_networkx_nodes(G, pos, node_color=model.labels_, cmap=plt.cm.Set1,node_size=10000)
nx.draw_networkx_edges(G, pos, edge_color='gray',width=2)
nx.draw_networkx_labels(G, pos, font_size=30, font_family='sans-serif')
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): w['weight'] for u, v, w in G.edges(data=True)}, font_size=30)
plt.axis('off')
plt.show()
