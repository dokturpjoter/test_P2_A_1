# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 11:26:06 2015

@author: Lipek
"""

###############################################################################
import os, sys
from multiprocessing import Pool

#sys.path.append(os.path.dirname(sys.argv[0]) + '/../../../../classes')
#sys.path.append(os.path.dirname(sys.argv[0]) + '/../../../classes')
sys.path.append(os.path.dirname(sys.argv[0]) + '/classes')

from Artificial_disease_1_2_3b_individual import Artificial_disease_1dot2_3 as ad

full_path = os.path.realpath(__file__)
models_path   = os.path.join(os.path.dirname(full_path) + '/../../models/')
data_path     = os.path.join(os.path.dirname(full_path) + '/../../data/')
mappings_path = os.path.join(os.path.dirname(full_path) + '/../../mappings/')
this_path = os.path.join(os.path.dirname(full_path) + '/')

###############################################################################
################################ SETUP ########################################
###############################################################################

simulation_data = {}

#################################### COMMON ###################################

simulation_data['stats_simple'] = True
simulation_data['make_z_scores'] = True

########################## COLLECTIVE EXPRESSION ##############################

simulation_data['shuffle'] = True
simulation_data['shuffle'] = False
simulation_data['shuffled_samples'] = 2

simulation_data['excluding'] = True
simulation_data['excluding'] = False
simulation_data['filter_grade'] = 11

simulation_data['fields'] = {'dis':'{disease} == 1'}

simulation_data['multiproc'] = False
#simulation_data['multiproc'] = True
simulation_data['processors'] = 32

############################ DISEASE PARAMS ###################################

disease_params = {}
disease_params['order'] = 1 #order of clustering neighborhood 
disease_params['patients_number'] = 200 # pn

"""
D_list     = [10, 50, 200] #disease related genes
S_list     = [10, 20, 50] #sick patiens number
rho_0_list = [0.005, 0.01, 0.05, 0.1] #expression density
A_list     = [0.0, 0.1, 0.5, 1.0] #disease expression density scaling
p_list     = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9] #clustering probability
"""

"""
# quick negative effect
D_list     = [20, 50, 100, 200] #disease related genes
S_list     = [10, 30] #sick patiens number
rho_0_list = [0.01, 0.02, 0.05] #expression density
#A_list     = [0.0, 0.1, 0.5, 1.0] #disease expression density scaling
#A_list     = [0.1, 0.2, 0.5, 1.0, 0.0] #disease expression density scaling
A_list     = [1.0, 0.5, 0.2, 0.0] #disease expression density scaling

#p_list     = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9] #clustering probability

p_list     = [0.9, 0.75, 0.5, 0.25, 0.1, 0.0] #clustering probability
"""

"""
# quick negative effect
D_list     = [50, 20, 100, 200] #disease related genes
S_list     = [30, 10] #sick patiens number
rho_0_list = [0.01, 0.02, 0.03, 0.04, 0.05] #expression density
#A_list     = [0.0, 0.1, 0.5, 1.0] #disease expression density scaling
#A_list     = [0.1, 0.2, 0.5, 1.0, 0.0] #disease expression density scaling
A_list     = [0.5, 0.2, 0.0] #disease expression density scaling

#p_list     = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9] #clustering probability

p_list     = [0.5, 0.1, 0.9, 0.0] #clustering probability
"""


# quick negative effect
D_list     = [50] #disease related genes
S_list     = [30] #sick patiens number
rho_0_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1] #expression density
rho_0_list = [0.01, 0.02] #expression density
#A_list     = [0.0, 0.1, 0.5, 1.0] #disease expression density scaling
#A_list     = [0.1, 0.2, 0.5, 1.0, 0.0] #disease expression density scaling
A_list     = [0.5] #disease expression density scaling
#p_list     = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9] #clustering probability
p_list     = [0.5] #clustering probability

# quick negative effect
D_list     = [50] #disease related genes
S_list     = [50] #sick patiens number
rho_0_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1] #expression density
rho_0_list = [0.01, 0.02] #expression density
#A_list     = [0.0, 0.1, 0.5, 1.0] #disease expression density scaling
#A_list     = [0.1, 0.2, 0.5, 1.0, 0.0] #disease expression density scaling
A_list     = [0.5] #disease expression density scaling
#p_list     = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9] #clustering probability
p_list     = [0.5] #clustering probability


# analogous to six
D_list     = [50] #disease related genes
S_list     = [100] #sick patiens number
rho_0_list = [0.01] #expression density
#A_list     = [0.0, 0.1, 0.5, 1.0] #disease expression density scaling
#A_list     = [0.1, 0.2, 0.5, 1.0, 0.0] #disease expression density scaling
A_list     = [0.5] #disease expression density scaling
#p_list     = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9] #clustering probability
p_list     = [0.5] #clustering probability


"""
# quick negative effect
D_list     = [50] #disease related genes
S_list     = [30] #sick patiens number
rho_0_list = [0.01] #expression density
#A_list     = [0.0, 0.1, 0.5, 1.0] #disease expression density scaling
A_list     = [0.25] #disease expression density scaling
p_list     = [0.75] #clustering probability

A_list     = [0.15] #disease expression density scaling
p_list     = [0.5] #clustering probability
"""
#####################################################
# analogous to six
D_list     = [50] #disease related genes
S_list     = [100] #sick patiens number
rho_0_list = [0.01, 0.02, 0.05] #expression density
rho_0_list = [0.05] #expression density
#A_list     = [0.0, 0.1, 0.5, 1.0] #disease expression density scaling
#A_list     = [0.1, 0.2, 0.5, 1.0, 0.0] #disease expression density scaling
A_list     = [0.5] #disease expression density scaling
#p_list     = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9] #clustering probability
p_list     = [0.25] #clustering probability



# analogous to six
D_list     = [50] #disease related genes
S_list     = [20] #sick patiens number
rho_0_list = [0.05] #expression density
#A_list     = [0.0, 0.1, 0.5, 1.0] #disease expression density scaling
#A_list     = [0.1, 0.2, 0.5, 1.0, 0.0] #disease expression density scaling
A_list     = [0.1] #disease expression density scaling
#p_list     = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9] #clustering probability
p_list     = [0.5] #clustering probability



# 3x3
# analogous to six
D_list     = [50] #disease related genes
S_list     = [20] #sick patiens number
rho_0_list = [0.01, 0.05, 0.1] #expression density
#A_list     = [0.0, 0.1, 0.5, 1.0] #disease expression density scaling
#A_list     = [0.1, 0.2, 0.5, 1.0, 0.0] #disease expression density scaling
A_list     = [0.5, 0.25, 0.1] #disease expression density scaling
#p_list     = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9] #clustering probability
p_list     = [0.5] #clustering probability


# 2x2
# analogous to six
D_list     = [50] #disease related genes
S_list     = [20] #sick patiens number
#rho_0_list = [0.01, 0.05, 0.1] #expression density
rho_0_list = [0.01, 0.05] #expression density
#A_list     = [0.0, 0.1, 0.5, 1.0] #disease expression density scaling
#A_list     = [0.1, 0.2, 0.5, 1.0, 0.0] #disease expression density scaling
#A_list     = [0.5, 0.25, 0.1] #disease expression density scaling
A_list     = [0.5, 0.25] #disease expression density scaling
#p_list     = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9] #clustering probability
p_list     = [0.5] #clustering probability





#####################################################
#simulation_data['shuffled_samples'] = 100
disease_params['realisations'] = 30

############################ NETWORK MODEL ####################################

#simulation_data['model'] = 'BA_1'
#simulation_data['model'] = 'BA_2'
simulation_data['model'] = 'BA_3'

#simulation_data['model'] = 'ER_0.001'
#simulation_data['model'] = 'ER_0.006'
#simulation_data['model'] = 'ER_0.01'
#simulation_data['model'] = 'ER_0.1'

#simulation_data['model'] = '2D'
#simulation_data['model']= 'CYC'

###############################################################################
############################### PRERUN ########################################
###############################################################################

if(simulation_data['model'] == 'BA_1'):
    simulation_data['network_type']  = 'BA'
    simulation_data['network_n'] = 1000
    simulation_data['network_m'] = 1
    simulation_data['model_prefix']  = 'BA_nn_' + str(simulation_data['network_n']) + '_nm_' + str(simulation_data['network_m']) 
    simulation_data['in_data_path']  = str(this_path) + 'out_BA_1/'
    simulation_data['out_data_path'] = str(this_path) + 'out_BA_1/'  
elif(simulation_data['model'] == 'BA_2'):
    simulation_data['network_type']  = 'BA'
    simulation_data['network_n'] = 1000
    simulation_data['network_m'] = 2
    simulation_data['model_prefix']  = 'BA_nn_' + str(simulation_data['network_n']) + '_nm_' + str(simulation_data['network_m']) 
    simulation_data['in_data_path']  = str(this_path) + 'out_BA_2/'
    simulation_data['out_data_path'] = str(this_path) + 'out_BA_2/'       
elif(simulation_data['model'] == 'BA_3'):
    simulation_data['network_type']  = 'BA'
    simulation_data['network_n'] = 1000
    simulation_data['network_m'] = 3
    simulation_data['model_prefix']  = 'BA_nn_' + str(simulation_data['network_n']) + '_nm_' + str(simulation_data['network_m']) 
    simulation_data['in_data_path']  = str(this_path) + 'out_BA_3/'    
    simulation_data['out_data_path'] = str(this_path) + 'out_BA_3/'
elif(simulation_data['model'] == 'ER_0.001'):
    simulation_data['network_type']  = 'ER'
    simulation_data['network_n'] = 1000
    simulation_data['network_p'] = 0.001 
    simulation_data['model_prefix']  = 'ER_nn_' + str(simulation_data['network_n']) + '_np_' + str(simulation_data['network_p']) 
    simulation_data['in_data_path']  = str(this_path) + 'out_ER_0.001/'
    simulation_data['out_data_path'] = str(this_path) + 'out_ER_0.001/'
elif(simulation_data['model'] == 'ER_0.006'):
    simulation_data['network_type']  = 'ER'
    simulation_data['network_n'] = 1000
    simulation_data['network_p'] = 0.006 
    simulation_data['model_prefix']  = 'ER_nn_' + str(simulation_data['network_n']) + '_np_' + str(simulation_data['network_p']) 
    simulation_data['in_data_path']  = str(this_path) + 'out_ER_0.006/'
    simulation_data['out_data_path'] = str(this_path) + 'out_ER_0.006/'
elif(simulation_data['model'] == 'ER_0.01'):
    simulation_data['network_type']  = 'ER'
    simulation_data['network_n'] = 1000
    simulation_data['network_p'] = 0.01 
    simulation_data['model_prefix']  = 'ER_nn_' + str(simulation_data['network_n']) + '_np_' + str(simulation_data['network_p']) 
    simulation_data['in_data_path']  = str(this_path) + 'out_ER_0.01/'
    simulation_data['out_data_path'] = str(this_path) + 'out_ER_0.01/'
elif(simulation_data['model'] == 'ER_0.1'):
    simulation_data['network_type']  = 'ER'
    simulation_data['network_n'] = 1000
    simulation_data['network_p'] = 0.1 
    simulation_data['model_prefix']  = 'ER_nn_' + str(simulation_data['network_n']) + '_np_' + str(simulation_data['network_p']) 
    simulation_data['in_data_path']  = str(this_path) + 'out_ER_0.1/'
    simulation_data['out_data_path'] = str(this_path) + 'out_ER_0.1/'
elif(simulation_data['model'] == '2D'):
    simulation_data['network_type']  = '2D'
    simulation_data['network_x'] = 40
    simulation_data['network_y'] = 25 
    simulation_data['model_prefix'] = '2D_nx_' + str(simulation_data['network_x']) + '_ny_' + str(simulation_data['network_y'])
    simulation_data['in_data_path']  = str(this_path) + 'out_2D/'
    simulation_data['out_data_path'] = str(this_path) + 'out_2D/'
elif(simulation_data['model'] == 'CYC'):
    simulation_data['network_type']  = 'CYC'
    simulation_data['network_n'] = 1000
    simulation_data['model_prefix'] = 'CYC_nn_' + str(simulation_data['network_n'])
    simulation_data['in_data_path']  = str(this_path) + 'out_CYC/'
    simulation_data['out_data_path'] = str(this_path) + 'out_CYC/'
elif(simulation_data['model'] == 'REG'):
    simulation_data['network_type']  = 'REG'
    simulation_data['network_n'] = 1000
    simulation_data['network_d'] = 10
    simulation_data['model_prefix'] = 'REG_nn_' + str(simulation_data['network_n']) + '_nd_' + str(simulation_data['network_d'])
    simulation_data['in_data_path']  = str(this_path) + 'out_REG/'
    simulation_data['out_data_path'] = str(this_path) + 'out_REG/' 

simulation_data['img'] = str(this_path) + 'img_4/' 

###############################################################################

whole_params_stack = []
                   
for D in D_list: 
    for S in S_list:
        #for rho_0 in rho_0_list: 
        for A in A_list:      
            for p in p_list:                     
                disease_params_2 = disease_params.copy()   
                disease_params_2['G']     = simulation_data['network_n'] # genes
                disease_params_2['D']     = D                                                             
                disease_params_2['S']     = S
                disease_params_2['rho_0_list'] = rho_0_list
                disease_params_2['A']     = A
                disease_params_2['p']     = p
                
                record = {}
                record['simulation_data'] = simulation_data
                record['disease_params']  = disease_params_2

                whole_params_stack.append(record)


"""
for D in D_list: 
    for S in S_list:
        for rho_0 in rho_0_list: 
            for A in A_list:      
                for p in p_list:                     
                    disease_params_2 = disease_params.copy()   
                    disease_params_2['G']     = simulation_data['network_n'] # genes
                    disease_params_2['D']     = D                                                             
                    disease_params_2['S']     = S
                    disease_params_2['rho_0'] = rho_0
                    disease_params_2['A']     = A
                    disease_params_2['p']     = p
                    
                    record = {}
                    record['simulation_data'] = simulation_data
                    record['disease_params']  = disease_params_2

                    whole_params_stack.append(record)
"""
###############################################################################
############################### RUN ###########################################
###############################################################################
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

add = ad()

res = 0.025


for simnum in range(0,20):
    plt.figure(figsize=(8,8))
    plot_no = 1
    
    for A in A_list:
        for rho_0 in rho_0_list:       
            whole_params = whole_params_stack[0]
            
            whole_params['disease_params']['A'] = A
            #whole_params['disease_params']['rho_0'] = rho_0
            whole_params['disease_params']['rho_0_list'] = [rho_0]
            
            
            E, G = add.simulate_ind(whole_params['simulation_data'], whole_params['disease_params'])  
            
            coherence_C = []
            coherence_S = []
            
            for i, expressed_genes in enumerate(E['expressed_genes_over']):
                g = G.copy()
                g = g.subgraph(expressed_genes)
                N = len(g.nodes())
                g.remove_nodes_from([ n for n,d in g.degree_iter() if d == 0 ])
                
                K = len(g.nodes())
                           
                if i <= S_list[0]:
                    if N > 0:
                        coherence_S.append(float(K)/float(N))
                    else:
                        coherence_S.append(0)
                else:
                    if N > 0:
                        coherence_C.append(float(K)/float(N))
                    else:
                        coherence_C.append(0)
                    
            plt.subplot(2,2,plot_no)
            #plt.subplot(3,3,plot_no)
            sns.kdeplot(np.array(coherence_C), shade=True, shade_lowest=False, alpha = 0.66, color="g", bw=res) 
            sns.kdeplot(np.array(coherence_S), shade=True, shade_lowest=False, alpha = 0.66, color="r", bw=res) 
            plt.title('A = ' +str(A)+'  $\\rho_0$ = ' +str(rho_0))
            
            if plot_no > 2:
                plt.xlabel('c', size=16)  
            
            plt.yticks([])   
            
            if plot_no == 1:
                plt.legend(['C - controls', 'P - patients']) 
            
            plot_no += 1
            
            #print whole_params['disease_params']
    plt.suptitle('p = '+ str(whole_params['disease_params']['p']),size=17) 
    plt.savefig('ind_hist_sc_2x2_6_p'+str(whole_params['disease_params']['p'])+'_i_'+str(simnum)+'.png', bbox_inches='tight', dpi=150)
    plt.savefig('ind_hist_sc_2x2_6_p'+str(whole_params['disease_params']['p'])+'_i_'+str(simnum)+'.pdf', bbox_inches='tight')
    #plt.tight_layout()
    plt.show() 
    plt.close()






  