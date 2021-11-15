%
%     This file is part of CasADi.
%
%     CasADi -- A symbolic framework for dynamic optimization.
%     Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl,
%                             K.U. Leuven. All rights reserved.
%     Copyright (C) 2011-2014 Greg Horn
%
%     CasADi is free software; you can redistribute it and/or
%     modify it under the terms of the GNU Lesser General Public
%     License as published by the Free Software Foundation; either
%     version 3 of the License, or (at your option) any later version.
%
%     CasADi is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
%     Lesser General Public License for more details.
%
%     You should have received a copy of the GNU Lesser General Public
%     License along with CasADi; if not, write to the Free Software
%     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
%

% An implementation of direct multiple shooting
% Joel Andersson, 2016

clear all; clc; close all;

addpath('utils\')
addpath('casadi\');
import casadi.*

%%
par.T = 0.1; % Time horizon
par.N_opt = 1; % number of control intervals
par.N_sim = 1000; % number of control intervals
par.m = 28*1e-3;
par.g = 9.81;
par.J = [16.571710, 0.830806, 0.718277;
    0.830806, 16.655602, 1.800197;
    0.718277, 1.800197, 29.261652]*1e-6;
par.tau_k = 0.005964552;
par.tau_b = 1.563383*1e-5;
par.l_x = 28.243*1e-3;
par.l_y = 28.325*1e-3;
par.range = [0.1; 0.1; 0.1; pi/4; pi/4; pi/4; 0.1; 0.1; 0.1; pi/4; pi/4; pi/4];

[model] = dynamics(par);
[nlp] = gen_nlp(model, par);
[tree] = lqr_tree(model, nlp, par);
%%
% load matlab.mat

% Simulation test
X = [];
U = [];
xk = (rand(2, 1)-0.5)*2;
[idx_set, idx_step] = match_lqr_tree(xk, X_opt);
X(:, 1) = xk;
for k=0:N*100-1
    idx = floor(k/100)+idx_step;
    if idx > N
        idx = N;
    end
    
    x_interp = X_opt{idx_set}(:, idx) + mod(k, 100)/100*(X_opt{idx_set}(:, idx+1) - X_opt{idx_set}(:, idx));
    if idx ~= N
        u_interp = U_opt{idx_set}(idx) + mod(k, 100)/100*(U_opt{idx_set}(idx+1) - U_opt{idx_set}(idx));
    else
        u_interp = U_opt{idx_set}(idx) + mod(k, 100)/100*(0 - U_opt{idx_set}(idx));
    end
    uk = u_interp - K_opt{idx_set}{idx}*(xk-x_interp);
    U(:, k+1) = uk;
    
    sol = dyn_sim('x0', xk, 'p', uk);
    xk = full(sol.xf);
    X(:, k+2) = xk;
end

tgrid = linspace((idx_step-1)/N*T, T, N+2-idx_step);
figure()
hold on
plot(tgrid, X_opt{idx_set}(1, idx_step:N+1), '--')
plot(tgrid, X_opt{idx_set}(2, idx_step:N+1), '-')
stairs(tgrid, [U_opt{idx_set}(idx_step:N); nan], '-.')
xlabel('t')
legend('x1','x2','u')

tgrid = linspace(0, T, N*100+1);
figure()
hold on
plot(tgrid, X(1, :), '--')
plot(tgrid, X(2, :), '-')
stairs(tgrid, [U, nan], '-.')
xlabel('t')
legend('x1','x2','u')

figure()
hold on
% for i = 1:length(X_opt)
%     plot(X_opt{i}(1, :), X_opt{i}(2, :), ':');
% end
plot(X_opt{idx_set}(1, idx_step:end), X_opt{idx_set}(2, idx_step:end), '--')
plot(X(1, :), X(2, :), '-')

function [x, xdot, tau] = quadrotor_dynamics(par)

import casadi.*

% Declare model variables
x = SX.sym('x', 12, 1);
tau = SX.sym('Y', 4, 1);

m = par.m;
g = par.g;
J = par.J;
[m, g, J, l_x, l_y, tau_k, tau_b] = deal(par.m, par.g, par.J, par.l_x, par.l_y, par.tau_k, par.tau_b);
[p_n, p_e, h, u, v, w, phi, theta, psi, p, q, r] = deal(x(1), x(2), x(3), x(4), x(5), x(6), x(7), x(8), x(9), x(10), x(11), x(12));
% [F, tau_phi, tau_theta, tau_psi] = deal(tau(1), tau(2), tau(3), tau(4));
[F, tau_phi, tau_theta, tau_psi] = deal(sum(tau), ...
[-l_x, -l_x, l_x, l_x]*tau, ...
[l_y, -l_y, -l_y, l_y]*tau, ...
[tau_k, -tau_k, tau_k, -tau_k]*tau);

% Model equations
R = [cos(psi), -sin(psi), 0;
    sin(psi), cos(psi), 0;
    0, 0, 1]*...
    [cos(theta), 0, sin(theta);
    0, 1, 0;
    -sin(theta), 0, cos(theta)]*...
    [1, 0, 0;
    0, cos(phi), -sin(phi);
    0, sin(phi), cos(phi)];
M = blkdiag(diag(repmat(m, 3, 1)), J);
h = [m*angvel2skew([p; q; r])*[u; v; w]; cross([p; q; r], J*[p; q; r])] + ...
    [R'*inv(diag([1;1;-1]))*[0; 0; m*g]; zeros(3, 1)];
Y = [0; 0; -F; tau_phi; tau_theta; tau_psi];
tmp = jacobian(reshape(R, 9, 1), [phi; theta; psi]);
tmp = [skew2angvel(R'*reshape(tmp(:, 1), 3, 3)), ...
    skew2angvel(R'*reshape(tmp(:, 2), 3, 3)), ...
    skew2angvel(R'*reshape(tmp(:, 3), 3, 3))];

xdot = [diag([1;1;-1])*R*[u; v; w]; tmp\[p; q; r]; M\(Y-h)];

end

function [model] = dynamics(par)

import casadi.*

dt_opt = par.T;
dt_sim = par.T/10;

[x, xdot, u] = quadrotor_dynamics(par);
xd = SX.sym('xd', 12, 1);

% Objective term
L = (x-xd)'*(x-xd) + u'*u;

% Continuous time Jacobian
A = jacobian(xdot, x);
B = jacobian(xdot, u);
model.jac = Function('jacobian', {x, u}, {A, B}, {'x', 'u'}, {'A', 'B'});

% CVODES from the SUNDIALS suite
dae = struct('x',x,'p',[u; xd],'ode',xdot,'quad',L);
opts = struct('tf',dt_opt);
model.dyn_opt = integrator('F', 'cvodes', dae, opts);
opts = struct('tf',dt_sim);
model.dyn_sim = integrator('F', 'cvodes', dae, opts);

end

function [nlp] = gen_nlp(model, par)

import casadi.*

dyn = model.dyn_opt;
N = par.N_opt;

% Start with an empty NLP
w={};
nlp.w0 = [];
nlp.lbw = [];
nlp.ubw = [];
J = 0;
g={};
nlp.lbg = [];
nlp.ubg = [];

% "Lift" initial conditions
Xd = MX.sym('Xd', 12);
X0 = MX.sym('X0', 12);
Xk = X0;

% Formulate the NLP
for k=0:N-1
    % New NLP variable for the control
    Uk = MX.sym(['U_' num2str(k)], 4);
    w = {w{:}, Uk};
    nlp.lbw = [nlp.lbw; repmat(-inf, 4, 1)];
    nlp.ubw = [nlp.ubw; repmat(inf, 4, 1)];
    nlp.w0 = [nlp.w0; repmat(0, 4, 1)];
    
    % Integrate till the end of the interval
    Fk = dyn('x0', Xk, 'p', [Uk; Xd]);
    Xk_end = Fk.xf;
    J = J+Fk.qf;
    
    % New NLP variable for state at end of interval
    Xk = MX.sym(['X_' num2str(k+1)], 12);
    w = [w, {Xk}];
    nlp.lbw = [nlp.lbw; repmat(-inf, 12, 1)];
    nlp.ubw = [nlp.ubw; repmat(inf, 12, 1)];
    nlp.w0 = [nlp.w0; repmat(0, 12, 1)];
    
    % Add equality constraint
    g = [g, {Xk_end-Xk}];
    nlp.lbg = [nlp.lbg; repmat(0, 12, 1)];
    nlp.ubg = [nlp.ubg; repmat(0, 12, 1)];
end

% Create an NLP solver
prob = struct('f', J, 'x', vertcat(w{:}), 'g', vertcat(g{:}), 'p', [X0; Xd]);
nlp.solver = nlpsol('solver', 'ipopt', prob);

end

function [tree] = lqr_tree(model, nlp, par)

N = par.N_opt;
jac = model.jac;
solver = nlp.solver;
w0 = nlp.w0;
lbw = nlp.lbw;
ubw = nlp.ubw;
lbg = nlp.lbg;
ubg = nlp.ubg;
range = par.range;

tree.X_opt = sparse(0, 0);
tree.U_opt = sparse(0, 0);
tree.K_opt = sparse(0, 0);
tree.P_opt = sparse(0, 0);
tree.parent = [];

x = zeros(12, 1);
u = repmat(par.m*par.g/4, 4, 1);
        
sol = jac('x', x, 'u', u);
A = full(sol.A);
B = full(sol.B);

[K, P, ~] = lqr(A, B, 1e1*eye(8), eye(4));

tree.X_opt = blkdiag(tree.X_opt, x);
tree.U_opt = blkdiag(tree.U_opt, u);
tree.K_opt = blkdiag(tree.K_opt, K);
tree.P_opt = blkdiag(tree.P_opt, P);
tree.parent = [tree.parent; -1];

for i = 1:10
    x = [zeros(3, 1); (rand(9, 1)-0.5)*2.*range(4:end)];
    tmp = repmat({x}, length(tree.parent), 1);
    tmp = blkdiag(tmp{:});
    cost = diag(tmp'*tree.P_opt*tmp);
    [cost, idx] = min(cost);
    xd = tree.X_opt(idx, (idx-1)*12+1:idx*12);
    
    while cost > 100
        xp = x;
        x = x/2;
        tmp = repmat({x}, length(tree.parent), 1);
        tmp = blkdiag(tmp{:});
        cost = diag(tmp'*tree.P_opt*tmp);
        [cost, idx] = min(cost);
        xd = tree.X_opt(idx, (idx-1)*12+1:idx*12);
    end
    x = xp;
    
    if cost > 100
       
        xd = tree.X_opt(idx, (idx-1)*12+1:idx*12);
        
        % Solve the NLP
        sol = solver('x0', w0, 'p', [x0; xd], 'lbx', lbw, 'ubx', ubw,...
            'lbg', lbg, 'ubg', ubg);
        w_opt = reshape(full(sol.x), 16, []);
        u = w_opt(1:4, :);
        
        sol = jac('x', x, 'u', u);
        A = full(sol.A);
        B = full(sol.B);
        
        [K, P, ~] = lqr(A, B, Q, R);
        
        tree.X_opt = blkdiag(tree.X_opt, x);
        tree.U_opt = blkdiag(tree.U_opt, u);
        tree.K_opt = blkdiag(tree.K_opt, K);
        tree.P_opt = blkdiag(tree.P_opt, P);
        tree.parent = [tree.parent; idx];
    end
end

% for i=0:grid^6-1
%     tmp = dec2base(i, grid);
%     x0 = zeros(6, 1);
%     for j=1:numel(tmp)
%         x0(numel(tmp)-j+1) = base2dec(tmp(j), grid);
%     end
%     
%     x0 = range([4:6, 10:12]).*((x0)/(grid-1)*2-1);
%     if all(round(x0, 3) == 0)
%         continue;
%     end
%     
%     x0 = [zeros(3, 1); x0(1:3); zeros(3, 1); x0(4:6)];
%     
%     % Solve the NLP
%     sol = solver('x0', w0, 'p', x0, 'lbx', lbw, 'ubx', ubw,...
%         'lbg', lbg, 'ubg', ubg);
%     w_opt = reshape(full(sol.x), 16, []);
%     
%     tree.X_opt{end+1} = [x0, w_opt(5:16, :)];
%     tree.U_opt{end+1} = w_opt(1:4, :);
%     tree.K_opt{end+1} = {};
%     
%     % Solve TVLQR
%     for k = 0:N-1
%         x = tree.X_opt{end}(k+1);
%         u = tree.U_opt{end}(k+1);
%         
%         sol = jac('x', x, 'u', u);
%         A = full(sol.A);
%         B = full(sol.B);
%         
%         [tree.K_opt{end}{k+1}, ~, ~] = lqr(A, B, eye(12), eye(4), zeros(12, 4));
%     end
%     
% end

end

function [idx_set, idx_step] = match_lqr_tree(x0, X_opt)
idx_set = 1;
idx_step = 1;
dis = 1e10;
for i = 1:length(X_opt)
    for j = 1:size(X_opt{i}, 2)
        if norm(x0 - X_opt{i}(:, j)) < dis
            idx_set = i;
            idx_step = j;
            dis = norm(x0 - X_opt{i}(:, j));
        end
    end
end
end
