clear
% Deal with data produced by GPU_GRFS code
infos = load('infos.inf');  PRECIS=infos(1); nx=infos(2); ny=infos(3); nz=infos(3);
if (PRECIS==8), DAT = 'double';  elseif (PRECIS==4), DAT = 'single';  end 
fid = fopen('RND_3D.res','rb'); D = fread(fid, inf, DAT, 0, 'n');  fclose(fid);
D   = reshape(D,[nx,ny,nz]);
% plot
figure(2),clf
slice(D,fix(nx/2),fix(ny/2),fix(nz/2)),shading flat,axis image,colorbar
drawnow

