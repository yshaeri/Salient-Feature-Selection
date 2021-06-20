import numpy as np;

def tree_test(data=None,labels=None,coeff=None,class_id=None):

    Label_set = np.unique(labels);
    Nclass= len(Label_set);
    depth = int(np.ceil(np.log2(Nclass)));
    
    idtest = np.array( [None]*len(data) );
    for i in range(len(data)):
        inode = 0;
        for idepth in range(1,depth+1):
            V = sum(np.concatenate([data[i,:],[1]])*coeff[idepth][inode]);
            if V >= 0:  child=0;
            else:       child=1;
            inode = 2*inode +child;

            if idepth == depth -1:
                if class_id[1][inode] != None:
                    idtest[i] = class_id[1][inode];
                    break;
            if idepth == depth:
                idtest[i] = class_id[2][inode];
    
    err = sum(labels != idtest) / len(labels);
    return err;