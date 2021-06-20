import numpy as np;
from Spike_Isolation.Event_Classification.Yang_NSP.dtree_spikes_master.node_split import node_split;
from Spike_Isolation.Event_Classification.Yang_NSP.dtree_spikes_master.impurity import impurity;

def tree_split(data=None,labels=None,bits=None,repeats=50):

# tree_split constructs a quantized oblique decison tree on the n by p 
# matrix data. Each row of data corresponds to a sample in p dimensional space.
# labels stores class labels for data in an n by 1 array. A class label
# should be an integer greater than zero. bits represents the resolution of
# coefficients for a hyperplane and should be an integer greater than zero.
# If bits is one, the function returns an axis-parallel decision tree.
# Otherwise, it returns an oblique decision tree with the specified quantization.

    # The function returns the tree structure stored in coeff with one by
# Ndepth cell array. Ndepth means the depth of the tree, equal to
# ceiling(log2(Nclass)), where Nclass means the number of classes in data. The
# ith cell contains 2^(i-1) elements. If the coefficients of an element are
# zero, the element represents a leaf node. The class label of a leaf node
# is stored in class_id as a 1 by 2 cell array. The first cell
# corresponds to the last but one depth with size of 2^(Ndepth-1)and the
# second cell corresponds the last depth with size of 2^Ndepth. A zero will
# be assigned if it is not a leaf node.

    # If a tree structure is like the diagram below, then coeff is a 1 by 3 array, such
# that coeff = [{N1}, {N2, N3}, {0, N4, 0, N5}].
# class_id = [{1, 0, 2, 0}, {0, 0, 3, 4, 0, 0, 5, 6}]
#            N1
#           / \
#         N2   N3
#        / \   / \
#       1  N4 2  N5
#         / \    / \
#        3   4  5   6
#
# For more detailed information about the algorithm for creating the oblique decision; # tree, please refer to
# S. K. Murthy, S. Kasif, and S. Salzberg, "A system for induction of
# oblique decision trees," J. Artif. Int. Res., vol. 2, pp. 1-32, 1994.
    if bits == 1:        opt=0;
    else:                opt=1;

    Label_set = np.unique(labels);
    Nclass = len(Label_set);
    depth  = int(np.ceil(np.log2(Nclass)));
    Ndim   = len(data[0,:]);
    N = len(data);
    attribute_min = np.min(data,axis=0) - 1;

    data_tree  = [data - [attribute_min]*N];
    labels_tree= [labels];
    impurity_node=[];
    coeff = {};
    #idepth,inode=1,0
    for idepth in range(1,depth+1):
        coeff[idepth] = {};
        if idepth <= depth -1:
            data_child = [];
            labels_child=[];
            for inode in range(0, 2**(idepth-1)):
                coeff[idepth][inode] = node_split(data_tree[inode],labels_tree[inode],opt,bits,repeats);
                Ndata = len(data_tree[inode]);

                V = data_tree[inode]*coeff[idepth][inode][:Ndim];
                V = sum(V.T) + coeff[idepth][inode][Ndim];

                idx=np.where(V > 0)[0];
                data_child  .append( data_tree  [inode][idx] );
                labels_child.append( labels_tree[inode][idx] );
                idx=np.where(V <=0)[0];
                data_child  .append( data_tree  [inode][idx] );
                labels_child.append( labels_tree[inode][idx] );

                if idepth == depth -1:
                    impurity_p, impurity_R, impurity_L = impurity(data_tree[inode],labels_tree[inode],coeff[idepth][inode]);
                    impurity_node.append(impurity_R);
                    impurity_node.append(impurity_L);

            data_tree  = data_child;
            labels_tree= labels_child;
            del(data_child,labels_child);
        elif idepth == depth:
            leafclass = 2**idepth -Nclass;
            class_id    = {1:[None]*(2**(idepth-1))};
            class_id[2] = [None]*(2**idepth);
            if leafclass != 0:
                impurity_depth,idx = sorted(impurity_node), np.argsort(impurity_node);
                for ileafclass in range(leafclass):
                    inode = idx[ileafclass];
                    _labels_tree = labels_tree[inode];
                    #class_cnts   = [ sum(_labels_tree==i) for i in set(_labels_tree) ];
                    class_cnts   = [ len(np.where(_labels_tree==i)[0]) for i in set(_labels_tree) ];
                    
                    #class_id[1][inode] = np.array(list(set(_labels_tree)))[np.where(class_cnts==max(class_cnts))[0]];
                    class_id[1][inode] = np.array(list(set(_labels_tree)))[np.argmax(class_cnts)];#??
                    coeff[idepth][inode] = np.array( [0]*(Ndim+1) );
                idx_split = list( set(range(2**(depth-1))) - set(idx[:leafclass]) )

                for isplit in range(2**(idepth-1)-leafclass):
                    inode = idx_split[isplit]
                    coeff[idepth][inode] = node_split( data_tree[inode],labels_tree[inode],opt,bits,repeats );
                    Ndata = len(data_tree[inode]);

                    V = data_tree[inode]*coeff[idepth][inode][:Ndim];
                    V = sum(V.T) + coeff[idepth][inode][Ndim];

                    idx = np.where(V > 0)[0];
                    labels_child = labels_tree[inode][idx];
                    #class_cnts   = [ sum(labels_child==i) for i in set(labels_child) ];
                    class_cnts   = [ len(np.where(labels_child==i)[0]) for i in set(labels_child) ];
                    #class_id[2][2*inode  ] = np.array(list(set(labels_child)))[np.where(class_cnts==max(class_cnts))[0]]
                    if(len(class_cnts)>0):
                        class_id[2][2*inode  ] = np.array(list(set(labels_child)))[np.argmax(class_cnts)];#??
                    idx = np.where(V <= 0)[0];
                    labels_child = labels_tree[inode][idx];
                    #class_cnts   = [ sum(labels_child==i) for i in set(labels_child) ];
                    class_cnts   = [ len(np.where(labels_child==i)[0]) for i in set(labels_child) ];
                    
                    #class_id[2][2*inode+1] = np.array(list(set(labels_child)))[np.where(class_cnts==max(class_cnts))[0]]
                    if(len(class_cnts)>0):
                        class_id[2][2*inode+1] = np.array(list(set(labels_child)))[np.argmax(class_cnts)];#??

            elif leafclass == 0:
                for isplit in range(2**(idepth-1)):
                    coeff[idepth][isplit] = node_split( data_tree[isplit],labels_tree[isplit],opt,bits,repeats);
                    Ndata = len(data_tree[isplit]);

                    V = data_tree[isplit]*coeff[idepth][isplit][:Ndim];
                    V = sum(V.T) + coeff[idepth][isplit][Ndim];

                    idx = np.where(V > 0)[0];
                    labels_child = labels_tree[isplit][idx];
                    #class_cnts   = [ sum(labels_child==i) for i in set(labels_child) ];
                    class_cnts   = [ len(np.where(labels_child==i)[0]) for i in set(labels_child) ];
                    #class_id[2][2*isplit  ] = np.array(list(set(labels_child)))[np.where(class_cnts==max(class_cnts))[0]]
                    if(len(class_cnts)>0):
                        class_id[2][2*isplit  ] = np.array(list(set(labels_child)))[np.argmax(class_cnts)];#??
                    idx = np.where(V <= 0)[0];
                    labels_child = labels_tree[isplit][idx];
                    #class_cnts   = [ sum(labels_child==i) for i in set(labels_child) ];
                    class_cnts   = [ len(np.where(labels_child==i)[0]) for i in set(labels_child) ];
                    #class_id[2][2*isplit+1] = np.array(list(set(labels_child)))[np.where(class_cnts==max(class_cnts))[0]]
                    if(len(class_cnts)>0):
                        class_id[2][2*isplit+1] = np.array(list(set(labels_child)))[np.argmax(class_cnts)];#??
        #
    ##
    coeff_unnormalize = {};
    for idepth in range(1,depth+1):
        coeff_unnormalize[idepth] = {};
        for inode in range(len(coeff[idepth])):
            coeff_unnormalize[idepth][inode] = coeff[idepth][inode].copy();
            if sum(coeff[idepth][inode] != 0) != 0:
                coeff_unnormalize[idepth][inode][Ndim] = coeff[idepth][inode][Ndim] - sum(coeff[idepth][inode][:Ndim]*attribute_min);

    coeff = coeff_unnormalize;
    return coeff, class_id;