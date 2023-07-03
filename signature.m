clc
clear all
close all

% Specify folder containing test images
folder1 = 'C:\Users\beste\Desktop\signature\signature_example\sign_data\sign_data\test\049\Dataset\X\';
% Specify folder containing groundtruth images
folder2 = 'C:\Users\beste\Desktop\signature\signature_example\sign_data\sign_data\test\049\Dataset\Groundtruth\';

new_directory1 = 'C:\Users\beste\Desktop\signature\signature_example\sign_data\sign_data\test\049\Dataset\CropImageFromTransaction';
if ~exist(new_directory1, 'dir')
    mkdir(new_directory1);
end

% Create new directory (if it doesn't exist)
new_directory2 = 'C:\Users\beste\Desktop\signature\signature_example\sign_data\sign_data\test\049\Dataset\Signature_detect';
if ~exist(new_directory2, 'dir')
    mkdir(new_directory2);
end

% Create new directory (if it doesn't exist)
new_directory3 = 'C:\Users\beste\Desktop\signature\signature_example\sign_data\sign_data\test\049\Dataset\Groudtruth_crop';
if ~exist(new_directory3, 'dir')
    mkdir(new_directory3);
end

% Get list of image file names in the folder
file_list1 = dir([folder1 '*.jpeg']);
% Get list of image file names in the folder
file_list2 = dir([folder2 '*.jpeg']);

% Loop over all images in the folder
for i = 1:length(file_list1)
    
    original_img = imread([folder1 file_list1(i).name]);
    g_img = imread([folder2,file_list2(i).name]);
    
    gray = rgb2gray(original_img);
    g_img = im2gray(g_img);

    binary = imbinarize(gray);
    g_img = imbinarize(g_img);

    inverted_binary = imcomplement(binary);

    [w1,h1] = size(inverted_binary);
    cropped1 = imcrop(inverted_binary, [floor(w1 * 0.05+1120) floor(h1 *0.24) w1 h1-1925]);
    [w2,h2] = size(g_img);
    cropped2 = imcrop(g_img, [floor(w2 * 0.05+1200) floor(h2 *0.24) w2 h2-1925]);
    
    se = strel('disk', 1);
    BW1 = imclearborder(cropped1);
    BW2 = imclose(BW1,se);
    BW3 = bwareaopen(BW2,1000);
    BW4 = imclearborder(cropped2);
    
    [row1, col1] = find(BW3);
    if ~isempty(row1) && ~isempty(col1)
        bbox1 = [min(col1), min(row1), max(col1)-min(col1), max(row1)-min(row1)];

        % Crop image from bounding box
        cropped_original_img_binary = imcrop(BW3, bbox1);
        cropped_original_img_edge = edge(cropped_original_img_binary,'canny');
    end

    [row2, col2] = find(BW4);
    if ~isempty(row2) && ~isempty(col2)
        bbox2 = [min(col2), min(row2), max(col2)-min(col2), max(row2)-min(row2)];

        % Crop image from bounding box
        cropped_g_img_binary = imcrop(BW4, bbox2);
        cropped_g_img_edge = edge(cropped_g_img_binary,'canny');
    end

    % Crop from original dataset
    % Get the bounding box coordinates in the original image
    bbox1_original = bbox1 + [floor(w1 * 0.05+1120), floor(h1 * 0.24), 0, 0];
    
    % Crop the signature from the original image using the bounding box coordinates
    crop_signature_original = imcrop(original_img, bbox1_original);

    % Save file to folder
    % Specify file name and full path for saving cropped image
    file_name = ['Crop_RGBimage_', num2str(i), '.jpeg'];
    full_path1 = fullfile(new_directory1, file_name);

    % Specify file name and full path for saving cropped image
    file_name2 = ['Crop_Binaryimage_', num2str(i), '.jpeg'];
    full_path2 = fullfile(new_directory2, file_name2);

    % Specify file name and full path for saving cropped image
    file_name3 = ['Crop_groundtruth_', num2str(i), '.jpeg'];
    full_path3 = fullfile(new_directory3, file_name3);

    % Check if file already exists
    if exist(full_path1, 'file')
        disp(['File ', file_name, ' already exists in directory. Skipping.'])
    else     
        % Save cropped image to folder
        imwrite(crop_signature_original, full_path1);
        disp(['File ', file_name, ' saved to directory.'])
    end

    % Check if file already exists
    if exist(full_path2, 'file')
        disp(['File ', file_name2, ' already exists in directory. Skipping.'])
    else     
        % Save cropped image to folder
        imwrite(cropped_original_img_binary, full_path2);
        disp(['File ', file_name2, ' saved to directory.'])
    end

    % Check if file already exists
    if exist(full_path3, 'file')
        disp(['File ', file_name3, ' already exists in directory. Skipping.'])
    else
        % Save cropped image to folder
        imwrite(cropped_g_img_binary, full_path3);
        disp(['File ', file_name3, ' saved to directory.'])
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Calculate global CoG
    [m1,n1] = size(cropped_original_img_edge);
    [X1,Y1] = meshgrid(1:n1,1:m1);
    global_cog_x1 = sum(sum(cropped_original_img_edge.*X1))/sum(sum(cropped_original_img_edge));
    global_cog_y1 = sum(sum(cropped_original_img_edge.*Y1))/sum(sum(cropped_original_img_edge));
    result1(i) = global_cog_x1;
    result2(i) = global_cog_y1;

    [m2,n2] = size(cropped_g_img_edge);
    size(cropped_g_img_edge);
    [X2,Y2] = meshgrid(1:n2,1:m2);
    global_cog_x2 = sum(sum(cropped_g_img_edge.*X2))/sum(sum(cropped_g_img_edge));
    global_cog_y2 = sum(sum(cropped_g_img_edge.*Y2))/sum(sum(cropped_g_img_edge));
    result3(i) = global_cog_x2;
    result4(i) = global_cog_y2;

    % Save variables to a file without overwriting
    save('Global_cog.mat', 'result1', 'result2', 'result3', 'result4');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Read Ground Truth image and calculate accuracy
    gt = cropped_g_img_edge;
    s_gt = regionprops(gt,'Area');
    sumGT = sum([s_gt.Area]);

    img = cropped_original_img_edge;
    s_img = regionprops(img,'Area');
    sumIMG = sum([s_img.Area]);
    
    % Calculate structural similarity index (SSIM)
    ssim_index = ssim(sumGT, sumIMG)*100;

    % Write accuracy to Excel file
    header = {'Row Number','Detect signature from Bank check','Detect signature from Groundtruth','SSIM Index'};
    save_dir = 'C:\Users\beste\Desktop\signature\signature_example\sign_data\sign_data\test\049\Dataset\result.xlsx';
    xlswrite(save_dir, header, 'Sheet1', 'A1');

    row = {num2str(i), file_list1(i).name, file_list2(i).name, ssim_index};
    row_num = num2str(i+1);
    save_dir = 'C:\Users\beste\Desktop\signature\signature_example\sign_data\sign_data\test\049\Dataset\result.xlsx';
    xlswrite(save_dir, row, 'Sheet1', ['A' row_num]);
end

load('Global_cog.mat');
save_dir = 'C:\Users\beste\Desktop\signature\signature_example\sign_data\sign_data\test\049\Dataset\distance.csv';

% Initialize table headers
header = {'File 1', 'File 2', 'Distance', 'Result'};

% Create empty table with headers
T = cell2table(cell(0, numel(header)), 'VariableNames', header);

% Loop through all combinations of files in file_list3 and file_list4
for i = 1:10
    for j = 1:10
        
        filename1 = sprintf('Crop_Binaryimage_%d.jpeg', i);
        filename2 = sprintf('Crop_groundtruth_%d.jpeg', j);

        % Compute distance between signatures
        distance = sqrt((result3(i) - result1(j))^2 + (result4(i) - result2(j))^2);

        % Check if signatures match
        if distance < 3
            message = "Signature match";
        else
            message = "Signature don't match";
        end

        % Add row to table
        row = {filename1, filename2, distance, message};
        T = [T; row];
    end
end

% Write table to CSV file
writetable(T, save_dir);
