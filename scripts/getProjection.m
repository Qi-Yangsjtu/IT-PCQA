function [ image, map, bb, data_remains, index, occupancy ] = getProjection( pt )


xyz = round(pt.Location);  rgb = single(pt.Color);

m2=max(xyz);
n2=min(xyz);
center_original = (m2+n2)/2;
xyz = xyz-(center_original);
m=max(xyz);
m2=max(m);
a=150/m2;
xyz = round(xyz*a)+151;
    
%% project to 6-faces of the bounding box in the principal component space
% Regularization
dataq = (xyz + 1);

data_remains = 0;
% xoy projection
dataq = [ dataq, rgb];                    % Preparation: xyz & color
[data_xoy, indexq_xoy] = sortrows(dataq, 3);                
[index1, index2, count_xoy] = getIndex(data_xoy(:,1:2));

index1 = indexq_xoy(index1);    index2 = indexq_xoy(index2); 
index_xoy = union(index1, index2);


    [data_yoz, indexq_yoz] = sortrows(dataq, 1);  
    [index3, index4, count_yoz] = getIndex(data_yoz(:,[2,3]));
    
    index3 = indexq_yoz(index3);    index4 = indexq_yoz(index4);
    index_yoz = union(index3, index4);
    
              
        [data_zox, indexq_zox] = sortrows(dataq,2); 
        [index5, index6, count_zox] = getIndex(data_zox(:,[3,1]));
        
        index5 = indexq_zox(index5);    index6 = indexq_zox(index6);
        index_zox = union(index5, index6);
        
        

            data_remains = dataq(setdiff(1:1:length(dataq), unique([index_xoy;index_yoz;index_zox])),:);
   

n=6;

datap1 = dataq;
datap2 = dataq(:,[2,3,1,4:6]);
datap3 = dataq(:,[3,1,2,4:6]);

% Decide number of the project plane
resolution = max(max(xyz))+1;
bb = cell(n,1);  index = cell(n,1);
map = cell(n,1);  image = cell(n,1);   occupancy = cell(n,1);
Y = cell(n,1);      U = cell(n,1);    V = cell(n,1);
%% Generation of depth maps & color images
for i = 1:n
    datap = eval(['datap',num2str(ceil(i/2))]);
    index{i} = eval(['index',num2str(i)]);
    bb{i} = [min(datap(:,1:2)), max(datap(:,1:2))];

    map{i} = zeros(resolution);
    map{i}(sub2ind(size(map{i}),datap(index{i},2),datap(index{i},1))) = datap(index{i},3);  % for quantization q2

    Y{i} = zeros(resolution); U{i} = zeros(resolution); V{i} = zeros(resolution);  
    Y{i}(sub2ind(size(Y{i}),datap(index{i},2),datap(index{i},1))) = datap(index{i},4);  % image axis and projected data axis mapping
    U{i}(sub2ind(size(U{i}),datap(index{i},2),datap(index{i},1))) = datap(index{i},5);
    V{i}(sub2ind(size(V{i}),datap(index{i},2),datap(index{i},1))) = datap(index{i},6);
    image{i} = (cat(3, Y{i}, U{i}, V{i}));
	occupancy{i} = map{i} & 1;
end
end