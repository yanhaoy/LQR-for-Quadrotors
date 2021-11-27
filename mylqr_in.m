x = ones(12, 1)*1e-3*0;
xd = ones(12, 1)*1e-3*0;
xd(1) = 0.1;
u = mylqr(x, xd);