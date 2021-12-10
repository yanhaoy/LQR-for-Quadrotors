x = ones(12, 1)*1e-3*0;
xd = ones(12, 1)*1e-3*0;
xd(1) = 0.1;
u = mylqr(x, xd);

x = ones(12, 1)*1e-3*0;
xd = ones(12, 1)*1e-3*0;
xd(1) = 0.1;
u = mylqr_pfl(x, xd);

x = ones(12, 1)*1e-3*0;
xd = ones(12, 1)*1e-3*0;
xd(1) = 0.1;
x(1:3) = rot_wb(x(1:3), x(4:6), true);