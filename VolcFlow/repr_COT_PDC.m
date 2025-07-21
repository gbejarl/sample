
%set(gcf, 'position', [528         556        1106         423])



% a2=subplot(2,2,2);
% set(gca, 'position', [0.7 0.525 0.3 0.45])
% 
% a4=subplot(2,2,4);
% set(gca, 'position', [0.7 0.025 0.3 0.45])
% 



% TT = T;
% TT(h<1e-6)=0;


% Uncomment below for height
im_h = ind2rgb(round(h*40*4), hot);
kk = ind2rgb(round(gradient(z+h)*4*4+32*4), gray);
k=kk(:,:,1); k(h>0.001)=0; kk(:,:,1)=k;
k=kk(:,:,2); k(h>0.001)=0; kk(:,:,2)=k;
k=kk(:,:,3); k(h>0.001)=0; kk(:,:,3)=k;


% Uncomment below for velocity
% im_h = ind2rgb(round(u*40*4), hot);
% kk = ind2rgb(round(gradient(z+u)*4*4+32*4), gray);
% k=kk(:,:,1); k(u>0.001)=0; kk(:,:,1)=k;
% k=kk(:,:,2); k(u>0.001)=0; kk(:,:,2)=k;
% k=kk(:,:,3); k(u>0.001)=0; kk(:,:,3)=k;

cla
image(x,y,kk+im_h)

originalXlim = xlim;
originalYlim = ylim;

% Read the shapefile
shapefile = '/Users/gustavo/Library/CloudStorage/GoogleDrive-gbejarlo@mtu.edu/My Drive/Michigan Tech/Lahar Project/GIS/Layers/CSBI/glacier.shp'; % Replace with your shapefile path
SS = shaperead(shapefile);

shapefile2 = '/Users/gustavo/Library/CloudStorage/GoogleDrive-gbejarlo@mtu.edu/My Drive/CSBI/Final/GIS/reprojected_seismic_stations.shp';
SS2 = readgeotable(shapefile2);

% Overlay the shapefile on the 2D plot
for k = 1:length(SS)
    % Extract the coordinates of the shapefile
    x_shp = SS(k).X;
    y_shp = SS(k).Y;

    % Ensure the shapefile data is properly clipped to remove NaNs
    x_shp(isnan(x_shp)) = [];
    y_shp(isnan(y_shp)) = [];

    % Plot the shapefile as a line
    plot(x_shp, y_shp, 'c', 'LineWidth', 1);
end

% Extract coordinates from the shapefile structure
lat = SS2.Shape.Y; % Latitude
lon = SS2.Shape.X; % Longitude

scatter(lon, lat, 50, 'filled', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'w');

% Loop through each point and add a label based on 'field_1'
for i = 1:height(SS2)
    % Extract the label text for each point
    labelText = string(SS2.field_1(i));  % Assuming 'field_1' is the field name
    
    % Display the label slightly offset from each point
    text(lon(i)+150, lat(i), labelText, 'FontSize', 8, 'VerticalAlignment', 'middle', 'HorizontalAlignment', 'left', 'Color', 'w');
end

shading interp
set(gca, 'ydir', 'normal')
set(gca, 'position', [0 0.125 0.8 0.8])
axis equal

hold on
contourf(x, y, -z, -[0.2 0.2], 'facecolor', [0.2 0.4 0.6]);

% xlabel('Easting'); ylabel('Northing'); 
dd = sprintf('Time: %5.1f minutes', t/60);
text(778800, 9929800, dd, 'fontweight', 'bold', 'fontsize', 12, 'color' ,'w','BackgroundColor', 'k', 'Margin', 3)
%view(11, 30)
hold on
set(gcf, 'color', 'w');
%set(gca, 'position', [0.05 0.05 0.45 1])
text(778800, 9929500, 'Simulated with VolcFlow (Kelfoun, 2009)', 'fontsize', 7, 'color' ,'w','BackgroundColor', 'k', 'Margin', 3)
text(778800, 9929200, 'CSBI Scenario Only', 'fontsize', 7, 'fontweight', 'bold', 'color' ,'r','BackgroundColor', 'k', 'Margin', 3)

contour(x, y, z, 100:100:600, 'color', 'k', 'linewidth', 0.5);

% Add colorbar with labels
if length(unique(h))>1
    colorbar;
    caxis([min(h(:)) max(h(:))]); % Adjust the color axis limits if needed
    colormap(hot);
    cb = colorbar;
    cb.Label.String = 'Flow height (m)';
end

% sget(gcf, 'Position');
% Set font size for axis ticks
set(gca, 'FontSize', 16);
set(gcf, 'position', [256,210,1000,700]);

% Reset axis limits to the original extent
xlim(originalXlim);
ylim(originalYlim);