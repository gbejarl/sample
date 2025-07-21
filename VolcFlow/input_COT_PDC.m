
%Uploads the topography
load('topo_totale_2.mat'); %issu de ExtractZones

[nrow, ncol]=size(z);

%Initial thickness - no thickness initially
h=z*0;

% PARAMETERS of the SIMULATION ----------------------
dt = 1;                 %initial time step
tstep_adjust = 1;       %1 = automatic adjustement of the time step
CFDcond      = 0.75;    %adjust dt according to velocity < 0.75 * dx/dt
dtmin        = 0.06;    %minimum time step
dtmax        = 10;      %maximum time step
dtplot       = 10;      %plot time step
tmax         = 1200;  %max time step of the simulation (but can be stopped manaully : stop button or ^C)
g            = 9.81;    %gravity

delta_int = 0/180*pi;   %internal friction angle
delta_bed = 0/180*pi;   %basal friction angle
cohesion  = 7500;       %yield strength for a Bingham flow (or cohesion for a Coulomb law)
rho       = 1500;       %density
coef_u2   = 0.01;       %Turbulent coefficient
viscosity = 0;       %viscosity in Pa s

erase_dat = 1;          %Erase the data-file if present 

UU=h*0;                 %added by the user to store the max velocity
%representation = 'cla; imagesc(h, [0 5]); hold on; contour(z, [1 1], ''w''); axis equal tight;'; %'repr_lava_large_fleches;     UU=max(UU, u );';   %Used to plot the data
representation = 'repr_COT_PDC;';
f_avi = 'cotopaxi_pdc_v.avi';  %Used to make a movie file
f_data = 'cotopaxi_pdc_v.dat'; %Used to store the data

%Used to compute the streamlines in repr_lava_large_fleches.m
%H=[0;0];T=0; XG=[]; YG=[];
%[xg, yg] = meshgrid(50:20:300, 50:20:650);

%initial velocity equals zero
% u_xx = z(1:nrow, 1:ncol-1)*0;
% u_xy = z(1:nrow, 1:ncol-1)*0;
% u_yy = z(1:nrow-1, 1:ncol)*0;
% u_yx = z(1:nrow-1, 1:ncol)*0;
% umx = h*0; umy = h*0;

%initial velocity equals something else
u0 = 30;  % Desired initial velocity in the x-direction
u_xx = u0 * ones(nrow, ncol-1);
u_xy = u0 * ones(nrow, ncol-1);
u_yy = u0 * ones(nrow-1, ncol);
u_yx = u0 * ones(nrow-1, ncol);
umx = h*0; umy = h*0;

%Uses the image fissure_large.tif to compute below the volume rate according to the total volume and the time of eruption chosen
ind_source = find(f<100);
t_erupt = 600; Volume = 9.75e6;
%Source conditions
source = 'if t<= t_erupt; h(ind_source) = h(ind_source) + dhS*dt; end;';
%Boundary conditions
bound_cond = 'h2(1,:)=0; h2(end,:)=0; h2(:,1)=0; h2(:,end)=0; h2(f==115)=0;   %uix = interp2(umx  ,xg,yg); uiy = interp2(umy, xg,yg); xg = xg + uix.*dt; yg = yg + uiy.*dt;';

%dimensions réelles des bords des cellules en fonction de la pente locale
dx_x = []; dx_y = []; dy_x = []; dy_y = [];
dx_x(1:nrow, 1:ncol-1)=dx_horiz; dy_x(1:nrow, 1:ncol-1)=dy_horiz; dy_y(1:nrow-1, 1:ncol)=dy_horiz; dx_y(1:nrow-1, 1:ncol)=dx_horiz;
dy_x(2:nrow-1, 1:ncol-1) = sqrt(  ( (z(3:nrow,1:ncol-1)+z(3:nrow,2:ncol))/2-(z(1:nrow-2,1:ncol-1)+z(1:nrow-2,2:ncol))/2 ).^2 + (2*dy_horiz).^2  ) /2;
dx_y(1:nrow-1, 2:ncol-1) = sqrt(  ( (z(1:nrow-1,3:ncol)+z(2:nrow,3:ncol))/2-(z(1:nrow-1,1:ncol-2)+z(2:nrow,1:ncol-2))/2 ).^2 + (2*dx_horiz).^2  ) /2;
dx_x(:,1:ncol-1) = sqrt(  ( z(:,2:ncol) - z(:,1:ncol-1) ).^2 + dx_horiz^2);
dy_y(1:nrow-1,:) = sqrt(  ( z(2:nrow,:) - z(1:nrow-1,:) ).^2 + dy_horiz^2);

%Surface of each mesh along the slope to compute the volume rate of each cell
S=[];
S(1:nrow, 1:ncol)=dx_horiz*dy_horiz;
S(2:nrow-1 , 2:ncol-1) = ( dx_x(2:nrow-1, 1:ncol-2)/2 + dx_x(2:nrow-1, 2:ncol-1)/2 ) .* ( dy_y(2:nrow-1, 2:ncol-1)/2 + dy_y(1:nrow-2, 2:ncol-1)/2 );
S(1,:)=S(2,:);S(nrow,:)=S(nrow-1,:);S(:,1)=S(1,2);S(:,ncol)=S(:, ncol-1);
surface_source = sum(sum(S(f<100))); 
dhS = Volume/surface_source/t_erupt;
