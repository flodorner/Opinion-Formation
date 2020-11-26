from model import weighted_balance_general


##two communities 
m_test= weighted_balance_general(d=5,n_vertices = 250, n_edges=500, phi=0.5,alpha=0.3,dist=1.0)
for i in range(40000):
    if i%2500==0:
        m_test.draw_graph(str(i))
        pass
    m_test.step()
    
##many communities
m_test= weighted_balance_general(d=5,n_vertices = 250, n_edges=500, phi=0.5,alpha=0.3,dist=0.6)
for i in range(40000):
    if i%100==0:
        m_test.draw_graph(str(i))
    m_test.step()
