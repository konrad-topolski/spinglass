
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from PIL import Image, ImageDraw   
import pycosat     
import time
import networkx as nx
from matplotlib import colors
import matplotlib.gridspec as gridspec
#from scipy.linalg import kron
#give J_couplings as a symmetric array
#give H_fields as a vector



    
def H_0(N):
    one=np.array([[1.,0],[0,1.]])
    sx=np.array([[0,1],[1,0]])
    spin_x_operators=np.zeros((N,2**N,2**N))
    for j in range(N):
        temp=1 #1-dimensional array = a scalar
        for k in range(N): 
            if k==j:
                temp=np.kron(sx,temp)
            else:
                temp=np.kron(one,temp)                       
        spin_x_operators[j,:,:]=temp
    H_0=-np.sum(spin_x_operators,axis=0)
    return H_0

def H_1(N,J_couplings,H_fields): #J_couplings is a symmetric matrix with zeros on the diagonal, H_fields is a vector
    sz=np.array([[1,0],[0,-1]])
    one=np.array([[1.,0],[0,1.]])
    spin_z_operators=np.zeros((N,2**N,2**N))
    for j in range(N):
        temp=1 #1-dimensional array = a scalar
        for k in range(N): 
            if k==j:
                temp=np.kron(sz,temp)
            else:
                temp=np.kron(one,temp)                       
        spin_z_operators[j,:,:]=temp   
    #J_couplings is a matrix NxN, each spin_z is a matrix 2**N x 2**N, there are N of them 
    #Jons=np.tensordot(J_couplings,spin_z_operators,axes=(1,0))
    #s_otimes_Jons=np.tensordot(spin_z_operators, Jons ,axes=(0,0))
    #Hpart1=-np.einsum('ijjk',s_otimes_Jons) #summing over 
    
    #alternatively, we can actually do this: take all tensor products, and then contract on the relevant indices
    J_otimes_s=np.tensordot(J_couplings,spin_z_operators,axes=0)
    print(J_otimes_s)
    s_otimes_J_otimes_s=np.tensordot(spin_z_operators, J_otimes_s, axes=0)
    Hpart1=-np.einsum('abcaddce',s_otimes_J_otimes_s)/2 #/2 because we included all combinations twice
    #print(Hpart1)
    Hpart2out=np.tensordot(H_fields,spin_z_operators,axes=0)
    Hpart2=-np.einsum('aabc',Hpart2out)
    
    #Htest1=np.zeros((2**N,2**N))
    #Htest2=np.zeros((2**N,2**N))
    #for i in range(N):
    #    for j in range(i):
    #        Htest1+=-J_couplings[i,j]*spin_z_operators[i,:,:]*spin_z_operators[j,:,:]
    #    Htest2+=-H_fields[i]*spin_z_operators[i,:,:]
    H_1=Hpart1+Hpart2    
    return H_1

def H_1_easier(N,J_couplings,H_fields): #J_couplings is a symmetric matrix with zeros on the diagonal, H_fields is a vector
    sz=np.array([[1,0],[0,-1]])
    one=np.array([[1.,0],[0,1.]])
    spin_z_operators=np.zeros((N,2**N,2**N))
    for j in range(N):
        temp=1 #1-dimensional array = a scalar
        for k in range(N): 
            if k==j:
                temp=np.kron(sz,temp)
            else:
                temp=np.kron(one,temp)                       
        spin_z_operators[j,:,:]=temp   
        
    Hpart1=np.zeros((2**N,2**N))
    Hpart2=np.zeros((2**N,2**N))
    for i in range(N):
        for j in range(N):
            Hpart1+=-J_couplings[i,j]*spin_z_operators[i,:,:]*spin_z_operators[j,:,:]
            
        Hpart2+=-H_fields[i]*spin_z_operators[i,:,:]
    Hpart1/=2    
    H_1=Hpart1+Hpart2    
    return H_1

def H_total(N,parameter,J_couplings,H_fields):
    return (1-parameter)*H_0(N)+parameter*H_1_easier(N,J_couplings,H_fields)

def energy_gap(N,parameter,J_couplings, H_fields): 
    eigenvalues, eigenvectors =  np.linalg.eigh(H_total(N,parameter,J_couplings, H_fields))
    #eigenvalues are real, so we can sort them:
    index=np.argsort(eigenvalues)
    energy_gap=eigenvalues[index[1]]-eigenvalues[index[0]]
    #index[0] is the index in the array eigenvalues corresponding to the 
    #print(eigenvectors)
    #v=eigenvectors[:,index[0]].T   #take the whole index[0]-th column, then transpose
    #print(v)
    return energy_gap 
def opt_runtime(N,J_couplings,H_fields):
    npoints=100
    range_param=np.linspace(0,0.999,npoints)
    energy_gaps=np.zeros(npoints)
    time=0
    for i in range(npoints):
        energy_gaps[i]=energy_gap(N,range_param[i],J_couplings,H_fields)
        time+=(range_param[1]-range_param[0])/(energy_gaps[i]**2)
    return time     

def expectation_values(N,parameter, J_couplings,H_fields):
    sz=np.array([[1,0],[0,-1]])
    one=np.array([[1.,0],[0,1.]])
    spin_z_operators=np.zeros((N,2**N,2**N))
    for j in range(N):
        temp=1 #1-dimensional array = a scalar
        for k in range(N): 
            if k==j:
                temp=np.kron(sz,temp)
            else:
                temp=np.kron(one,temp)                       
        spin_z_operators[j,:,:]=temp   
        
    eigenvalues, eigenvectors =  np.linalg.eigh(H_total(N,parameter,J_couplings, H_fields))
    index=np.argsort(eigenvalues)
    v=eigenvectors[:,index[0]]
    expectations_values=np.zeros(N)
    expectation_values= np.tensordot(v.T,np.tensordot(spin_z_operators,v,axes=(1,0)),axes=(0,1))
    return expectation_values
    #def entanglement_entropy(N,parameter, J_couplings,H_fields):

def pictures_evolve(N,J_couplings,H_fields):
    npoints=100
    energy_gaps=np.zeros(npoints)
    spin_values=np.zeros((npoints,N))
    range_param=np.linspace(0,0.995,npoints)
    range_param2=np.linspace(0.05,1,npoints)

    
    for i in range(npoints):
        energy_gaps[i]=energy_gap(N,range_param[i],J_couplings,H_fields)
        #if i%2==0:
        spin_values[i,:]=expectation_values(N,range_param[i], J_couplings,H_fields)
        #else:
        #spin_values[i,:]=expectation_values(N,range_param2[i], J_couplings,H_fields)

        
    fig=plt.figure(figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(range_param,energy_gaps,color="r")
    plt.title("Energy gap as a function of parameter \u03BB.",fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig("Energy_gap="+str(N)+".png")
    plt.clf()
    
    fig=plt.figure(figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
    
    for i in range(N):
        #if i%2==0:
        plt.plot(range_param,spin_values[:,i],label='Spin '+str(i+1))
        #else:
            #plt.scatter(range_param,spin_values[:,i],label='Spin '+str(i+1),marker='*',s=80,alpha=0.5)

    plt.title("Spins' expectation values as functions of parameter \u03BB.",fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc="upper left",fontsize=15)
    plt.savefig("new_random_spins_N="+str(N)+".png")
    plt.clf()
    
    return 1
    
def plot_energy_gap(N,J_couplings, H_fields): 
    npoints=100
    range_param=np.linspace(0,1,100)
    energygaps=np.zeros(npoints)
    for i in range(npoints):
        energygaps[i]=energy_gap(N,range_param[i],J_couplings, H_fields)
    plt.plot(range_param,energygaps)
    
def network_evolve(N,J_couplings,H_fields):
    
    npoints=100
    param_range=np.linspace(0,1,npoints)
    g=nx.Graph()
    for i in range(N):
        g.add_node(i)
        
    for i in range(N):
        for j in range(N):
            if(J_couplings[i,j]!=0):
                g.add_edge(i,j)
                g[i][j]["coupling"]=J_couplings[i,j]
                
    pos_fix = nx.spring_layout(g,k=0.2)
    #print(J_couplings)
    cumulative_spin=np.zeros((npoints,N))  
    for i in range(npoints):
        plt.clf()
        plt.close()
        gs = gridspec.GridSpec(8, 7)
        fig = plt.figure(figsize=(30,30)) 
        ax1 = plt.subplot(gs[0:5,0:5])
        plt.suptitle("Evolution of spins, parameter \u03BB="+str("%.2f" % param_range[i]),fontsize=30)
        cumulative_spin[0,:]=0
        #plt.subplot(221)# draw_networkx_nodes versus nx.draw
            # draw_networkx_nodes versus nx.draw
        for j in range(N):
            g.nodes[j]["spin"]=expectation_values(N,param_range[i], J_couplings,H_fields)[j]
        cumulative_spin[i,:]=expectation_values(N,param_range[i], J_couplings,H_fields)

        color_the_nodes=[(g.nodes[i]["spin"]*np.heaviside(g.nodes[i]["spin"],0.5),0,np.abs(g.nodes[i]["spin"])*(np.heaviside(-g.nodes[i]["spin"],0.5))) for i in range(N)]
        color_the_edges=['r' if J_couplings[i,j]>0 else 'b' for i in range(N) for j in range(i)]
        width_of_edges=[2+8*abs(J_couplings[i,j]) for i in range(N) for j in range(i)]
        nx.draw(g,pos=pos_fix,node_color=color_the_nodes, 
                node_size=[2000+2000*abs(g.nodes[i]["spin"]) for i in range(N) ],
               edge_color=color_the_edges,
               width=width_of_edges,with_labels=True, font_size=40,font_color='w',ax=ax1)
        
        #plt.subplot(222)
        ax2=plt.subplot(gs[0:4,5:8])
        ax2.set_title("Couplings",fontsize=35)
        cmap = plt.cm.seismic
        norm = colors.Normalize(vmin=-1, vmax=1)
        jcoupl=ax2.imshow(J_couplings,cmap='seismic',vmin=np.amin(J_couplings),vmax=np.amax(J_couplings))

        ax2.tick_params(axis='both', labelsize=30)
        cbar = fig.colorbar( jcoupl, cmap='seismic', norm = norm)
        cbar.ax.tick_params(labelsize=40)
                            # plt.colors.Normalize(vmin=5, vmax=10)
        
 
        #plt.colorbar(plt.cm.ScalarMappable( cmap=cmap), orientation='horizontal', label='Some Units')
        ax3=plt.subplot(gs[4:5,5:8])
        #ax3.H_fields,cmap='seismic')
        ax3.text(0.2,2.,s="H fields:",size=30)
        for l in range(H_fields.size):
            ax3.text(0.2,1.75-l*0.175,s="h_"+str(l)+"="+str("%.2f" % H_fields[l]),size=30)
        plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=())
        plt.xticks(fontsize=0)
        plt.yticks(fontsize=0)
        #plt.imshow(J_couplings,cmap='seismic',vmin=-1,vmax=1)
        #ax4=fig.add_subplot(323)
        #plt.plot(np.sin(np.linspace(0,10,100)))

        #plt.
        ax4=plt.subplot(gs[5:-1,:])
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.xlim(0,1)
        plt.ylim(-1,1)
        for k in range(N):
            ax4.plot(param_range[:i],cumulative_spin[:i,k],label="Spin "+str(k),linewidth=5)
        plt.legend(loc='upper left', fontsize=20)
        #ax4=fig.add_subplot(223)
        #plt.figure(figsize=(8,8))
        #plt.subplot(3,2,1)
        #plt.subplot(3,2,3)
        #plt.subplot(3,2,5)
        #plt.subplot(2,2,2)
        #plt.subplot(2,2,4)
        #plt.imshow(J_couplings,cmap='seismic')
        plt.savefig("evolution_new"+str(i)+".png")
        plt.show()
    return 1 

N=10
J_couplings=np.zeros((N,N))
J_couplings[0,1]=-1
J_couplings[1,2]=-1
J_couplings[2,3]=-1
J_couplings[3,7]=-1
J_couplings[6,7]=-1
J_couplings[5,6]=-1
J_couplings[4,5]=-1
J_couplings[0,4]=-1
J_couplings[0,5]=-1
J_couplings[1,5]=-1
J_couplings[1,6]=-1
J_couplings[2,6]=-1
J_couplings[2,7]=-1
J_couplings=J_couplings+J_couplings.T
H_fields=-0.5*np.ones(N)
N=8


#print(opt_runtime(N,J_couplings,H_fields))
N=10
#plots=10
#fig=plt.figure(figsize=(20,20))
#for i in range(plots):
    #J_couplings=(2*np.random.random_sample(size=N**2)-1).reshape(N,N)    
    #J_couplings=(J_couplings+J_couplings.T)/2
    #J_couplings=J_couplings-np.eye(N)*np.diag(J_couplings)
    #print(J_couplings)
    #H_fields=2*np.random.random_sample(size=N)-1
    #plot_energy_gap(N,J_couplings,H_fields)
J_couplings=(2*np.random.random_sample(size=N**2)-1).reshape(N,N)    
J_couplings=(J_couplings+J_couplings.T)/2
J_couplings=J_couplings-np.eye(N)*np.diag(J_couplings)  
H_fields=2*np.random.random_sample(size=N)-1
network_evolve(N,J_couplings,H_fields)
#pictures_evolve(N,J_couplings,H_fields)    
#plt.xticks(size=25)
#plt.yticks(size=25)
#plt.ylabel("Value of the energy gap", fontsize=25)
#plt.xlabel("ParametWer \u03BB", fontsize=25)
#plt.title("Evolutions of the energy gap for various uniform realizations.",fontsize=25)
#plt.savefig("random_uniform_energy_gaps_N="+str(N)+".png")
#ig.rcParams(figsize=(20,20))
#plt.show()    


# In[ ]:


J_couplings=np.array([[0,-1.2,-2.1],[-1.2,0,-3.8],[-2.1,-3.8,0]]) 
H_fields=np.array([0.5,0.,0.])
N=3

npoints=100
energy_gaps=np.zeros(npoints)
spin_values=np.zeros((npoints,N))
range_param=np.linspace(0,1,npoints)
for i in range(npoints):
    energy_gaps[i]=energy_gap(N,range_param[i],J_couplings,H_fields)
    spin_values[i,:]=expectation_values(N,range_param[i], J_couplings,H_fields)
fig=plt.figure(figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
plt.plot(range_param,energy_gaps,color="r")
plt.title("Energy gap as a function of parameter \u03BB.",fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.clf()
fig=plt.figure(figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
for i in range(N):
    plt.plot(range_param,spin_values[:,i],label='Spin '+str(i+1))
plt.title("Spins' expectation values as functions of parameter \u03BB.",fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc="upper left",fontsize=15)
#plt.savefig("spins.png")
plt.clf()
#plt.savefig("energy_gap.png")
#plt.show()

