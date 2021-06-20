function [coeff impurity_plane]=perturb(data,labels,coeff,opt);
N=length(data);
DIM=size(data,2);
V=data.*repmat(coeff(1:DIM),N,1);
V=(sum(V'))'+coeff(end);
class=unique(labels);
Nclass=length(class);
Pstag=1;
if opt==0
    dim_start=DIM+1;
else
    dim_start=1;
end

for dim=dim_start:DIM+1
    if dim==DIM+1
        U=V;
    else
        U=V./data(:,dim);
    end
    
    candidates=coeff(dim)-U;
    [candidates idx]=sort(candidates);
    labels_sort=labels(idx);
 
    for i = 1:Nclass
        class_logic(i, :) = labels_sort == class(i);
    end

    class_sum = cumsum(class_logic, 2);
    Pro1 = class_sum(:, 1:N-1) ./ repmat([1:N-1], Nclass, 1);
    Pro2 = [repmat(class_sum(:, N), 1, N-1) - class_sum(:, 1:N-1)] ./ repmat([N-1:-1:1], Nclass, 1);

    idx = find(Pro1 == 0);
    Pro1(idx) = 0.00001;
    idx = find(Pro2 == 0);
    Pro2(idx) = 0.00001;
    impurity_coeff = -[1:N-1]/N.*sum(Pro1.*log2(Pro1)) - [N-1:-1:1]/N.*sum(Pro2.*log2(Pro2));
    
    [min_impurity_coeff idx]=min(impurity_coeff);
    
    coeff_new=coeff;
    coeff_new(dim)=(candidates(idx)+candidates(idx+1))/2;
    
    impurity_plane=impurity(data,labels,coeff);
    impurity_plane_new=impurity(data,labels,coeff_new);
    if impurity_plane-impurity_plane_new>0.001
        coeff=coeff_new;
        impurity_plane=impurity_plane_new;
        Pstag=1;
    else
        if Pstag>rand(1) %-> what is for
            coeff=coeff_new;
            impurity_plane=impurity_plane_new;
        end
        Pstag=Pstag-0.1*Pstag;
    end
end
