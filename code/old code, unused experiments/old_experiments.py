def experiment_holme_N25():
    n_iterations = 10
    for n_vertices in [25]:
        for n_opinions in [5]:
            for phi in [0.05,0.5,0.95]:
                kw={"n_vertices":n_vertices, "n_opinions":n_opinions,"phi":phi}
                print(kw)
                loop = ("n_edges",np.arange(1,101*(n_vertices/25)**2,4*(n_vertices/25)**2,dtype=np.int))
                output = experiment_loop(kw,loop,metrics=metrics,n=n_iterations,model_type="Holme")
                median_plus_percentile_plot(output["variation"][1],output["sd_size_connected_component"])
                plt.title("Sd of community size for N={},ϕ={},|O|={} and varying M".format(n_vertices,phi,n_opinions))
                plt.xlabel("Number of Edges")
                plt.savefig(image_folder+"sd_25_N{}_ϕ{}_O{}".format(n_vertices,int(phi*100),n_opinions))
                plt.close()
                median_plus_percentile_plot(output["variation"][1],output["mean_size_connected_component"])
                plt.title("Mean of community size for N={},ϕ={},|O|={} and varying M".format(n_vertices,phi,n_opinions))
                plt.xlabel("Number of Edges")
                plt.savefig(image_folder+"mean_25_N{}_ϕ{}_O{}".format(n_vertices,int(phi*100),n_opinions))
                plt.close()
                median_plus_percentile_plot(output["variation"][1],output["time_to_convergence"])
                plt.title("Time steps to convergence for N={},ϕ={},|O|={} and varying M".format(n_vertices,phi,n_opinions))
                plt.xlabel("Number of Edges")
                plt.savefig(image_folder+"t_25_N{}_ϕ{}_O{}".format(n_vertices,int(phi*100),n_opinions))
                plt.close()


def experiment_WBT25():
    n_iterations = 10
    kw={}
    print(kw)
    loop = ("n_vertices",np.arange(2,25,4,dtype=np.int))
    model_type="Weighted Balance"
    output = experiment_loop(kw,loop,metrics=metrics,n=n_iterations,model_type=model_type)

    median_plus_percentile_plot(output["variation"][1],output["sd_size_connected_component"])
    plt.title("Sd of community size")
    plt.xlabel("Number of Edges")
    plt.savefig(image_folder+"sd_25")
    plt.close()
    median_plus_percentile_plot(output["variation"][1],output["mean_size_connected_component"])
    plt.title("Mean of community size")
    plt.xlabel("Number of Edges")
    plt.savefig(image_folder+"mean_25")
    plt.close()
    median_plus_percentile_plot(output["variation"][1],output["time_to_convergence"])
    plt.title("Time steps to convergence")
    plt.xlabel("Number of Edges")
    plt.savefig(image_folder+"t_25")
    plt.close()

def dynamics_graph():
    "for gifs in the presentation. 2D network evolution"
    n_vertices = 25
    m= weighted_balance_general(d=2,n_vertices = n_vertices,
                                n_edges=n_vertices*2, phi=0.52,alpha=0.3,dist=0.4 ) #f=lambda x:x
    k=1
    ts=0
    #while m.convergence() == False:
    while ts<100:
        res.add_op_mat(m)
        
        
         
        fig=plt.figure(figsize=(4,4))
        ax=fig.add_axes([0.15, 0.1, 0.8, 0.8])
        pos={i:m.vertices[i] for i in range(len(m.vertices)) }
        nx.draw(m.graph,pos,ax=ax,node_size=30,alpha=0.5,node_color="blue", with_labels=False)
        limits=plt.axis('on')
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        plt.title("t={}".format(ts))
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        
        plt.savefig('general graph {:03d}.jpg'.format(ts))
        plt.close()
        # how many steps/rounds per image
        for j in range(2):
            ts=ts+1
            for i in range(n_vertices):
                
                m.step()
        
        k=k+1
        print(k)

### method for model.py as alternative convergence criterium
"""
self.vertices_previous = np.copy(self.vertices)
    def has_changed_more_than(self,threshold):
        '''as an alternative / proxy to computing connected components,
        check if opinions changed significantly since the last time this function was called'''
        if  np.sum(np.abs(self.vertices_previous - self.vertices))<threshold*self.n_vertices: 
            return False
        else: 
            self.vertices_previous = np.copy(self.vertices)
            return True

has_changed=A.has_changed_more_than(1e-2)
                    if done == has_changed:
                        print("done is not close"+str(done)+str(has_changed))
"""