function [index1, index2,count] = getIndex(data)

[~, ia1, ic1] = unique(data,'rows','first','legacy');    %  2. pick up min/max depth with equal xy
[~, ia2, ~] = unique(data,'rows','last','legacy'); 
count = hist(ic1,unique(ic1));

index1 = ia1;
index2 = ia2; 

end

