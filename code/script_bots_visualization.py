from model import weighted_balance_bots
from matplotlib import pyplot as plt
import os
import numpy as np
import networkx as nx


# Some of these visualizations were used for Figure 22, some for the gifs in the presentation.
# runtime ~10-40 min
# Creates A LOT of png images
#############################################
# Recommended to run within separate folder #
#############################################

image_folder=""

A=weighted_balance_bots(n_vertices=500, d=2, z=0.01, f=lambda x: x, alpha=0.5, n_edges=None,initial_graph=None,neutral_bots=False,both_sides=False,bot_positions=None, n_bots=0,epsilon=0,phi=0,connect=None,seeking_bots=False)
for i in range(25):
    for j in range(500):
        A.step()
    plt.scatter(A.vertices[:,0],(A.vertices[:,1]))
    plt.ylim(-1.1,1.1)
    plt.xlim(-1.1, 1.1)
    plt.savefig(image_folder+"gif_"+"id_nobots"+str(i))
    plt.close()


A=weighted_balance_bots(n_vertices=500, d=2, z=0.01, f=lambda x: np.sign(x)*np.abs(x)**0.5, alpha=0.5, n_edges=None,initial_graph=None,neutral_bots=False,both_sides=False,bot_positions=None, n_bots=0,epsilon=0,phi=0,connect=None,seeking_bots=False)
for i in range(25):
    for j in range(500):
        A.step()
    plt.scatter(A.vertices[:,0],(A.vertices[:,1]))
    plt.ylim(-1.1,1.1)
    plt.xlim(-1.1, 1.1)
    plt.savefig(image_folder+"gif_"+"nobots_eval_extreme"+str(i))
    plt.close()

A=weighted_balance_bots(n_vertices=500, d=2, z=0.01, f=lambda x: x, alpha=0.5, n_edges=None,initial_graph=None,neutral_bots=False,both_sides=False,bot_positions=None, n_bots=50,epsilon=0,phi=0,connect=None,seeking_bots=False)
for i in range(25):
    for j in range(500):
        A.step()
    plt.scatter(A.vertices[:,0],(A.vertices[:,1]))
    plt.ylim(-1.1,1.1)
    plt.xlim(-1.1, 1.1)
    plt.savefig(image_folder+"gif_"+"id_singlebots_50"+str(i))
    plt.close()

A=weighted_balance_bots(n_vertices=500, d=2, z=0.01, f=lambda x: x, alpha=0.5, n_edges=None,initial_graph=None,neutral_bots=False,both_sides=False,bot_positions=None, n_bots=5,epsilon=0,phi=0,connect=None,seeking_bots=False)
for i in range(25):
    for j in range(500):
        A.step()
    plt.scatter(A.vertices[:,0],(A.vertices[:,1]))
    plt.ylim(-1.1,1.1)
    plt.xlim(-1.1, 1.1)
    plt.savefig(image_folder+"gif_"+"id_singlebots_5"+str(i))
    plt.close()



A=weighted_balance_bots(n_vertices=500, d=2, z=0.01, f=lambda x: np.sign(x)*np.abs(x)**0.5, alpha=0.5, n_edges=None,initial_graph=None,neutral_bots=True,both_sides=False,bot_positions=None, n_bots=5,epsilon=0,phi=0,connect=None,seeking_bots=False)
for i in range(25):
    for j in range(500):
        A.step()
    plt.scatter(A.vertices[:,0],(A.vertices[:,1]))
    plt.ylim(-1.1,1.1)
    plt.xlim(-1.1, 1.1)
    plt.savefig(image_folder+"gif_"+"eval_extreme_5neutral"+str(i))
    plt.close()

A=weighted_balance_bots(n_vertices=500, d=2, z=0.01, f=lambda x: np.sign(x)*np.abs(x)**0.5, alpha=0.5, n_edges=None,initial_graph=None,neutral_bots=True,both_sides=False,bot_positions=None, n_bots=100,epsilon=0,phi=0,connect=None,seeking_bots=False)
for i in range(25):
    for j in range(500):
        A.step()
    plt.scatter(A.vertices[:,0],(A.vertices[:,1]))
    plt.ylim(-1.1,1.1)
    plt.xlim(-1.1, 1.1)
    plt.savefig(image_folder+"gif_"+"eval_extreme_100neutral"+str(i))
    plt.close()


A=weighted_balance_bots(n_vertices=500, d=2, z=0.01, f=lambda x: x, alpha=0.5, n_edges=499,initial_graph="barabasi_albert",neutral_bots=False,both_sides=False,bot_positions="bottom", n_bots=50,epsilon=0,phi=0,connect=None,seeking_bots=False)
for i in range(25):
    for j in range(500):
        A.step()
    pos = {i: A.vertices[i][:-1] for i in range(len(A.vertices))}
    nx.draw(A.graph, pos, node_size=30, alpha=0.5, node_color="blue", with_labels=False)
    plt.ylim(-1.1,1.1)
    plt.xlim(-1.1, 1.1)
    plt.savefig(image_folder+"gif_"+"id_bots_edge_bottom"+str(i))
    plt.close()


A=weighted_balance_bots(n_vertices=500, d=2, z=0.01, f=lambda x: x, alpha=0.5, n_edges=599,initial_graph="barabasi_albert",neutral_bots=False,both_sides=False,bot_positions=None, n_bots=50,epsilon=0,phi=0,connect=None,seeking_bots=False)
for i in range(25):
    for j in range(500):
        A.step()
    pos = {i: A.vertices[i][:-1] for i in range(len(A.vertices))}
    nx.draw(A.graph, pos, node_size=30, alpha=0.5, node_color="blue", with_labels=False)
    plt.ylim(-1.1,1.1)
    plt.xlim(-1.1, 1.1)
    plt.savefig(image_folder+"gif_"+"id_bots_edge"+str(i))
    plt.close()

A=weighted_balance_bots(n_vertices=500, d=2, z=0.01, f=lambda x: x, alpha=0.5, n_edges=599,initial_graph="barabasi_albert",neutral_bots=False,both_sides=False,bot_positions="top", n_bots=50,epsilon=0,phi=0,connect=None,seeking_bots=False)
for i in range(25):
    for j in range(500):
        A.step()
    pos = {i: A.vertices[i][:-1] for i in range(len(A.vertices))}
    nx.draw(A.graph, pos, node_size=30, alpha=0.5, node_color="blue", with_labels=False)
    plt.ylim(-1.1,1.1)
    plt.xlim(-1.1, 1.1)
    plt.savefig(image_folder+"gif_"+"id_bots_edge_top"+str(i))
    plt.close()



A=weighted_balance_bots(n_vertices=500, d=2, z=0.01, f=lambda x: x, alpha=0.5, n_edges=499,initial_graph="barabasi_albert",neutral_bots=False,both_sides=False,bot_positions=None, n_bots=0,epsilon=1,phi=0.5,connect=None,seeking_bots=False)
for i in range(50):
    for j in range(500):
        A.step()
    pos = {i: A.vertices[i][:-1] for i in range(len(A.vertices))}
    nx.draw(A.graph, pos, node_size=30, alpha=0.5, node_color="blue", with_labels=False)
    plt.ylim(-1.1,1.1)
    plt.xlim(-1.1, 1.1)
    plt.savefig(image_folder+"gif_"+"id_edge_eps1_"+str(i))
    plt.close()
    


A=weighted_balance_bots(n_vertices=500, d=2, z=0.01, f=lambda x: x, alpha=0.5, n_edges=499,initial_graph="barabasi_albert",neutral_bots=False,both_sides=False,bot_positions=None, n_bots=50,epsilon=1,phi=0.5,connect=None,seeking_bots=False)
for i in range(50):
    for j in range(500):
        A.step()
    pos = {i: A.vertices[i][:-1] for i in range(len(A.vertices))}
    nx.draw(A.graph, pos, node_size=30, alpha=0.5, node_color="blue", with_labels=False)
    plt.ylim(-1.1,1.1)
    plt.xlim(-1.1, 1.1)
    plt.savefig(image_folder+"gif_"+"id_edge_eps1_bots"+str(i))
    plt.close()


A=weighted_balance_bots(n_vertices=500, d=2, z=0.01, f=lambda x: x, alpha=0.5, n_edges=499,initial_graph="barabasi_albert",neutral_bots=False,both_sides=False,bot_positions=None, n_bots=50,epsilon=1,phi=0.5,connect=None,seeking_bots=True)
for i in range(50):
    for j in range(500):
        A.step()
    pos = {i: A.vertices[i][:-1] for i in range(len(A.vertices))}
    nx.draw(A.graph, pos, node_size=30, alpha=0.5, node_color="blue", with_labels=False)
    plt.ylim(-1.1,1.1)
    plt.xlim(-1.1, 1.1)
    plt.savefig(image_folder+"gif_"+"id_edge_eps1_bots_seeking"+str(i))
    plt.close()


A = weighted_balance_bots(n_vertices=500, d=2, z=0.01, f=lambda x: x, alpha=0.5, n_edges=499,
                          initial_graph="barabasi_albert", neutral_bots=False, both_sides=False, bot_positions=None,
                          n_bots=200, epsilon=1, phi=0.5, connect=None, seeking_bots=False)
for i in range(50):
    for j in range(500):
        A.step()
    pos = {i: A.vertices[i][:-1] for i in range(len(A.vertices))}
    nx.draw(A.graph, pos, node_size=30, alpha=0.5, node_color="blue", with_labels=False)
    plt.ylim(-1.1, 1.1)
    plt.xlim(-1.1, 1.1)
    plt.savefig(image_folder+"gif_"+ "id_edge_eps1_200bots" + str(i))
    plt.close()

A = weighted_balance_bots(n_vertices=500, d=2, z=0.01, f=lambda x: x, alpha=0.5, n_edges=499,
                          initial_graph="barabasi_albert", neutral_bots=False, both_sides=None, bot_positions="top",
                          n_bots=200, epsilon=1, phi=0.5, connect=None, seeking_bots=True)
for i in range(50):
    for j in range(500):
        A.step()
    pos = {i: A.vertices[i][:-1] for i in range(len(A.vertices))}
    nx.draw(A.graph, pos, node_size=30, alpha=0.5, node_color="blue", with_labels=False)
    plt.ylim(-1.1, 1.1)
    plt.xlim(-1.1, 1.1)
    plt.savefig(image_folder+"gif_"+ "id_edge_eps1_bots_seeking_200bots" + str(i))
    plt.close()

A=weighted_balance_bots(n_vertices=500, d=2, z=0.01, f=lambda x: np.sign(x)*np.abs(x)**0.5, alpha=0.5, n_edges=499,initial_graph="barabasi_albert",neutral_bots=False,both_sides=False,bot_positions=None, n_bots=0,epsilon=0.6,phi=0.5,connect=None,seeking_bots=False)
for i in range(50):
    for j in range(500):
        A.step()
    pos = {i: A.vertices[i][:-1] for i in range(len(A.vertices))}
    nx.draw(A.graph, pos, node_size=10, alpha=0.5, node_color="blue", with_labels=False)
    plt.ylim(-1.1,1.1)
    plt.xlim(-1.1, 1.1)
    plt.savefig(image_folder+"gif_"+"eval_extreme_edge_eps06"+str(i))
    plt.close()


A = weighted_balance_bots(n_vertices=500, d=2, z=0.01, f=lambda x: np.sign(x)*np.abs(x)**0.5, alpha=0.5, n_edges=499,
                          initial_graph="barabasi_albert", neutral_bots=True, both_sides=False, bot_positions=None,
                          n_bots=200, epsilon=0.6, phi=0.5, connect=None, seeking_bots=False)
for i in range(50):
    for j in range(500):
        A.step()
    pos = {i: A.vertices[i][:-1] for i in range(len(A.vertices))}
    nx.draw(A.graph, pos, node_size=10, alpha=0.5, node_color="blue", with_labels=False)
    plt.ylim(-1.1, 1.1)
    plt.xlim(-1.1, 1.1)
    plt.savefig(image_folder+"gif_"+ "eval_extreme_edge_eps06_200neutral" + str(i))
    plt.close()

A = weighted_balance_bots(n_vertices=500, d=2, z=0.01, f=lambda x: np.sign(x)*np.abs(x)**0.5, alpha=0.5, n_edges=499,
                          initial_graph="barabasi_albert", neutral_bots=True, both_sides=False, bot_positions=None,
                          n_bots=200, epsilon=0.6, phi=0.5, connect=None, seeking_bots=True)
for i in range(50):
    for j in range(500):
        A.step()
    pos = {i: A.vertices[i][:-1] for i in range(len(A.vertices))}
    nx.draw(A.graph, pos, node_size=10, alpha=0.5, node_color="blue", with_labels=False)
    plt.ylim(-1.1, 1.1)
    plt.xlim(-1.1, 1.1)
    plt.savefig(image_folder+"gif_"+ "eval_extreme_edge_eps06_200neutral_seeking" + str(i))
    plt.close()
