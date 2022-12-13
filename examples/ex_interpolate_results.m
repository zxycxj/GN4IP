% Clear all;
clear; clc; close all;

% Load the data
load("ex_interpolate_results.mat", "mesh_points", "mesh_data", "mesh_data_2", "grid_points", "grid_data")

% Plot the data
m = min(4, size(mesh_data,1));
for i = 1:m
    subplot(3,m,i); scatter(mesh_points(1,:), mesh_points(2,:), [], squeeze(mesh_data(i,1,:))); axis image; axis off;
    title("Sample "+num2str(i))
    subplot(3,m,i+m); imagesc(flipud(squeeze(grid_data(i,1,:,:)))); axis image; axis off;
    subplot(3,m,i+m+m); scatter(mesh_points(1,:), mesh_points(2,:), [], squeeze(mesh_data_2(i,1,:))); axis image; axis off;
end

% Add some labels
subplot(3,m,1); ylabel("Original on Mesh"); ax = gca; set(get(ax,'YLabel'),'Visible','on');
subplot(3,m,1+m); ylabel("Interp to Grid"); ax = gca; set(get(ax,'YLabel'),'Visible','on');
subplot(3,m,1+m+m); ylabel("Interp Back to Mesh"); ax = gca; set(get(ax,'YLabel'),'Visible','on');