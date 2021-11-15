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

addpath('casadi\');
import casadi.*

T = 10; % Time horizon
N = 20; % number of control intervals

% Declare model variables
x1 = SX.sym('x1');
x2 = SX.sym('x2');
x = [x1; x2];
u = SX.sym('u');

% Model equations
xdot = [(1-x2^2)*x1 - x2 + u; x1];

% Objective term
L = x1^2 + x2^2 + u^2;

% Continuous time dynamics
f = Function('f', {x, u}, {xdot, L});

% Continuous time Jacobian
A = jacobian(xdot, x);
B = jacobian(xdot, u);
jac = Function('jacobian', {x, u}, {A, B}, {'x', 'u'}, {'A', 'B'});

% CVODES from the SUNDIALS suite
dae = struct('x',x,'p',u,'ode',xdot,'quad',L);
opts = struct('tf',T/N);
F = integrator('F', 'cvodes', dae, opts);
opts = struct('tf',T/N/100);
F2 = integrator('F2', 'cvodes', dae, opts);

% Evaluate at a test point
Fk = F('x0',[0.2; 0.3],'p',0.4);
disp(Fk.xf)
disp(Fk.qf)

% Start with an empty NLP
w={};
w0 = [];
lbw = [];
ubw = [];
J = 0;
g={};
lbg = [];
ubg = [];

% "Lift" initial conditions
Xk = MX.sym('X0', 2);
w = {w{:}, Xk};
lbw = [lbw; 0; 1];
ubw = [ubw; 0; 1];
w0 = [w0; 0; 1];

% Formulate the NLP
for k=0:N-1
    % New NLP variable for the control
    Uk = MX.sym(['U_' num2str(k)]);
    w = {w{:}, Uk};
    lbw = [lbw; -1];
    ubw = [ubw;  1];
    w0 = [w0;  0];

    % Integrate till the end of the interval
    Fk = F('x0', Xk, 'p', Uk);
    Xk_end = Fk.xf;
    J=J+Fk.qf;

    % New NLP variable for state at end of interval
    Xk = MX.sym(['X_' num2str(k+1)], 2);
    w = [w, {Xk}];
    lbw = [lbw; -0.25; -inf];
    ubw = [ubw;  inf;  inf];
    w0 = [w0; 0; 0];

    % Add equality constraint
    g = [g, {Xk_end-Xk}];
    lbg = [lbg; 0; 0];
    ubg = [ubg; 0; 0];
end

% Create an NLP solver
prob = struct('f', J, 'x', vertcat(w{:}), 'g', vertcat(g{:}));
solver = nlpsol('solver', 'ipopt', prob);

% Solve the NLP
sol = solver('x0', w0, 'lbx', lbw, 'ubx', ubw,...
            'lbg', lbg, 'ubg', ubg);
w_opt = full(sol.x);

% Plot the solution
x1_opt = w_opt(1:3:end);
x2_opt = w_opt(2:3:end);
u_opt = w_opt(3:3:end);
tgrid = linspace(0, T, N+1);
figure()
hold on
plot(tgrid, x1_opt, '--')
plot(tgrid, x2_opt, '-')
stairs(tgrid, [u_opt; nan], '-.')
xlabel('t')
legend('x1','x2','u')

% Solve TVLQR
K = {};
for k=0:N-1
    x = [x1_opt(k+1); x2_opt(k+1)];
    u = u_opt(k+1);

    sol = jac('x', x, 'u', u);
    A = full(sol.A);
    B = full(sol.B);
    
    [K{k+1}, ~, ~] = lqr(A, B, eye(2), eye(1), zeros(2, 1));
end

% Simulation test
X = [];
U = [];
xk = [0; 1] + 0.1*rand(2, 1);
X(:, 1) = xk;
for k=0:N*100-1
    uk = u_opt(floor(k/100)+1) - K{floor(k/100)+1}*(xk-[x1_opt(floor(k/100)+1); x2_opt(floor(k/100)+1)]);
    U(:, k+1) = uk;
    
    sol = F2('x0', xk, 'p', uk);
    xk = full(sol.xf);
    X(:, k+2) = xk;
end

tgrid = linspace(0, T, N*100+1);
figure()
hold on
plot(tgrid, X(1, :), '--')
plot(tgrid, X(2, :), '-')
stairs(tgrid, [U, nan], '-.')
xlabel('t')
legend('x1','x2','u')