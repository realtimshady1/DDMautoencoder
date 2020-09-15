clc

% NC file path
fprintf('Displaying user dialog to select NetCDF file to inspect\n');
[~,ncfile,~] = uigetfile('*.nc', 'MERRByS L1b data');
ncinfoDDMs = ncinfo(ncfile);

% Create DDM dataset
if ~exist('DDM')
    DDM = [];
end

% Extract a 5000 dataset
fprintf('Extracting DDM Data...\n');
for i=1000:6000
    groupname =  sprintf('000%d/DDM', i);
    myvar = ncread(ncfile, groupname);
    DDM = cat(3, DDM, myvar);
end

% Save the DDM
save('DDMtrain', 'DDM');
fprintf('Done!\n');