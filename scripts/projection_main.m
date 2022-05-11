p =  genpath('/data/datasets/SJTU-PCQA/database/'); % root path for point cloud data
save_path1 = '/data/datasets/SJTU-PCQA/projection/projection1/'; % saving path for projection images of 6 perpendicular planes
save_path2 = '/data/datasets/SJTU-PCQA/projection/projection2/';
save_path3 = '/data/datasets/SJTU-PCQA/projection/projection3/';
save_path4 = '/data/datasets/SJTU-PCQA/projection/projection4/';
save_path5 = '/data/datasets/SJTU-PCQA/projection/projection5/';
save_path6 = '/data/datasets/SJTU-PCQA/projection/projection6/';
mkdir(save_path1);
mkdir(save_path2);
mkdir(save_path3);
mkdir(save_path4);
mkdir(save_path5);
mkdir(save_path6);
length_p = size(p,2);
path = {};
temp = [];
for i = 1:length_p 
    if p(i) ~= ':'
        temp = [temp p(i)];
    else 
        temp = [temp '/']; 
        path = [path ; temp];
        temp = [];
    end
end  
size(path)
file_num = size(path,1);
for i = 1:file_num
    file_path =  path{i}; 
    img_path_list = dir(strcat(file_path,'*.ply'));
    img_num = length(img_path_list);
    if img_num > 0
        for j = 1:img_num
            image_name = img_path_list(j).name;
            fprintf('%d %d %s\n',i,j,strcat(file_path,image_name));
            temp2 = pcread(strcat(file_path,image_name));
            [image,map,bb,data_remains,index,occupancy] = getProjection(temp2);
            onlyName = strrep(image_name,'.ply','');
            foldname=strsplit(file_path,'/');

            filename2 = strcat(save_path1,onlyName,'.png');
            imwrite(uint8(image{1}),filename2);
            filename2 = strcat(save_path2,onlyName,'.png');
            imwrite(uint8(image{2}),filename2);
            filename2 = strcat(save_path3,onlyName,'.png');
            imwrite(uint8(image{3}),filename2);
            filename2 = strcat(save_path4,onlyName,'.png');
            imwrite(uint8(image{4}),filename2);
            filename2 = strcat(save_path5,onlyName,'.png');
            imwrite(uint8(image{5}),filename2);
            filename2 = strcat(save_path6,onlyName,'.png');
            imwrite(uint8(image{6}),filename2);
        end
    end
end