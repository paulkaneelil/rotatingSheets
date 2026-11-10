%% Code to reproduce numerical results for the following paper:
% Paul Kaneelil, Sifan Yin, L. Mahadevan, "Patterns in soft spinning sheets
% and skirts"

% written with the help of Claude Opus 5.5 and GPT-6 Luna

% Critical whirling spin So_c(Gr/eps) using
%   * the FULL linearised shallow-shell (DMV) coupling (build_shell_operator_full),
%   * the LARGE-So base state: centrifugal stress N_cf (nonlinear stretching
%     neglected)
%
% Run the script: the first cell computes So_c(Gr/eps) (theory curve of
% Fig. 3b, before the x5.5 scaling applied for comparison with experiment);
% the second cell reproduces SI Fig. S1.
% Requires MATLAB R2018b or later (local functions in scripts, xline).
% Optional: distinguishable_colors (MATLAB File Exchange) for plot colors.
% Grid convergence: N = 301 -> 501 changes So_c by < 0.3% up to Gr/eps = 1e9.

results = sweep_critical_spin_cf_fullshell_main;


%% SI Fig. S1: modal projections of the operator terms (full shell, N_cf base)
% Run this cell on its own to reproduce the SI figure
% alpha = 0.2, nu = 0.5, m = 2 as in the main-text theory curve.
figS1 = make_fig_S1_operator_terms([1e3 1e4 1e5], 0.20, 0.50, 2, 301, 20001);

%%
function results = sweep_critical_spin_cf_fullshell_main()

    %% ---------------- user parameters ----------------
    nu      = 0.50;
    alpha   = 0.2; % 
    m       = 2;
    N       = 301;                 % radial grid for the eigenvalue problem (151 gives spurious modes at large Gr/eps)
    Nsag    = 20001;               % fine grid for the sag problem (bending layer ~ So^(-1/2))
    X_vals  = logspace(0, 9, 20);  % X = Gr/epsilon
    So_scan = logspace(0, 7.5, 60);% scan grid used to bracket the onset (searched from the top down)
    run_shell_selftest = true;

    if run_shell_selftest
        test_shell_operator_full(nu, m, alpha);
    end

    So_c    = nan(size(X_vals));
    lam_neg = false(size(X_vals));

    fprintf('\n  Gr/eps        So_c\n');
    for i = 1:numel(X_vals)
        X   = X_vals(i);
        res = @(S0) whirl_residual(S0, X, alpha, nu, m, N, Nsag);

        % Scan DOWNWARD from large So (residual < 0 above onset) to the first
        % So where the residual is positive. Searching from above avoids the
        % very-low-So region at large Gr/eps, where the sag is enormous and the
        % discretisation produces spurious modes.
        Sgrid = fliplr(So_scan);
        k = []; f_prev = res(Sgrid(1));
        if f_prev > 0
            warning('Residual positive at the largest scanned So for Gr/eps = %.3e', X);
            continue
        end
        for j = 2:numel(Sgrid)
            f_j = res(Sgrid(j));
            if f_j > 0, k = j; break; end
        end
        if isempty(k)
            warning('No onset bracket found for Gr/eps = %.3e', X);
            continue
        end
        So_c(i) = fzero(res, [Sgrid(k) Sgrid(k-1)], optimset('TolX', 1e-6));
        [~, lmin] = whirl_residual(So_c(i), X, alpha, nu, m, N, Nsag);
        lam_neg(i) = lmin < 0;
        if lam_neg(i)
            warning('Negative eigenvalue at the root for Gr/eps = %.3e: base state unstable', X);
        end
        fprintf('%10.3e  %11.4e\n', X, So_c(i));
    end

    %% ---------------- fits ----------------
    ok  = isfinite(So_c);
    big = ok & X_vals >= 1e5;                      % gravity-dominated branch
    p   = polyfit(log10(X_vals(big)), log10(So_c(big)), 1);
    local_slope = diff(log10(So_c(ok)))./diff(log10(X_vals(ok)));
    fprintf('\nLarge-Gr/eps fit (Gr/eps >= 1e5): So_c = %.3g (Gr/eps)^%.3f\n', 10^p(2), p(1));
    fprintf('Flat-plate plateau (smallest Gr/eps): So_c = %.3f\n', So_c(find(ok,1)));

    results.X_vals = X_vals;            results.So_c = So_c;
    results.fit_prefactor = 10^p(2);    results.fit_exponent = p(1);
    results.local_slope = local_slope;  results.lambda_negative = lam_neg;
    results.alpha = alpha; results.nu = nu; results.m = m; results.N = N; results.Nsag = Nsag;

    %% ---------------- plots ----------------
    figure('Color','w');
    subplot(1,2,1)
    loglog(X_vals, So_c, 'ko', 'MarkerFaceColor','w', 'LineWidth',1.5); hold on
    loglog(X_vals, 10^p(2)*X_vals.^(2/3), 'k--');
    xlabel('$Gr/\epsilon$','Interpreter','latex'); ylabel('$So_c^{\rm w}$','Interpreter','latex');
    title('full shell, $N_{\rm cf}$ base','Interpreter','latex'); grid on; box on
    subplot(1,2,2)
    semilogx(X_vals, So_c./X_vals.^(2/3), 'ko', 'MarkerFaceColor','w', 'LineWidth',1.5);
    xlabel('$Gr/\epsilon$','Interpreter','latex');
    ylabel('$So_c^{\rm w}/(Gr/\epsilon)^{2/3}$','Interpreter','latex'); grid on; box on

    assignin('base', 'cf_fullshell_results', results);
end


%% ========================================================================
function [res, lmin] = whirl_residual(S0, X, alpha, nu, m, N, Nsag)
% Residual  lambda_min - m^2*So  (positive below onset, negative above).
    [rs, df, d2f] = sag_shape_cf(S0, alpha, nu, Nsag);
    r    = linspace(alpha, 1, N).';
    dw0  = (X/S0)*interp1(rs, df,  r, 'spline');
    d2w0 = (X/S0)*interp1(rs, d2f, r, 'spline');
    [C1, C2] = cf_constants(alpha, nu);
    Nr   = S0*(C1 + C2./r.^2 - (3+nu)/8*r.^2);          % S0 * N_cf
    dNr  = S0*(-2*C2./r.^3 - (3+nu)/4*r);
    lmin = lowest_eigenvalue(r, dw0, d2w0, Nr, dNr, S0, nu, m);
    res  = lmin - m^2*S0;
end


function [C1, C2] = cf_constants(alpha, nu)
% Centrifugal stress N_cf = C1 + C2/r^2 - (3+nu)/8 r^2 (see SI)
    C2 = (alpha^2/8)*(1-nu)*((3+nu) - (1+nu)*alpha^2)/((1+nu) + (1-nu)*alpha^2);
    C1 = (3+nu)/8 - C2;
end


function [r, df, d2f] = sag_shape_cf(S0, alpha, nu, Nsag)
% Linear sag on the centrifugal stress:
%   (1/S0) lap_0^2 f - div(Ncf grad f) = 1,
%   clamp (r = alpha): f = f' = 0;  free (r = 1): f'' + nu f' = 0, (lap_0 f)' = 0.
% The left side is (1/r) d/dr[ r((1/S0)(lap_0 f)' - Ncf f') ]. Integrating once
% and using the free-edge conditions (shear-free and Ncf(1) = 0, i.e. vertical
% force balance) gives an exact 2nd-order problem for g = f':
%   (1/S0)(g'' + g'/r - g/r^2) - Ncf g = (r^2 - 1)/(2r),
%   g(alpha) = 0,   g'(1) + nu g(1) = 0.
% This is far better conditioned than discretising the 4th-order equation.
    r  = linspace(alpha, 1, Nsag).';
    dr = r(2) - r(1);
    [C1, C2] = cf_constants(alpha, nu);
    Nr = C1 + C2./r.^2 - (3+nu)/8*r.^2;
    e  = ones(Nsag,1);
    lo = (1/S0)*(1/dr^2 - 1./(2*dr*r));       % coefficient of g(i-1)
    di = (1/S0)*(-2/dr^2 - 1./r.^2) - Nr;     % coefficient of g(i)
    up = (1/S0)*(1/dr^2 + 1./(2*dr*r));       % coefficient of g(i+1)
    A  = spdiags([[lo(2:end); 0] di [0; up(1:end-1)]], -1:1, Nsag, Nsag);
    b  = (r.^2 - 1)./(2*r);
    A(1,:) = 0; A(1,1) = 1; b(1) = 0;                                   % g(alpha) = 0
    A(end,:) = 0; A(end,end-2:end) = [1 -4 3]/(2*dr); A(end,end) = A(end,end) + nu;
    b(end) = 0;                                                         % g'(1) + nu g(1) = 0
    df = A\b;                                                           % f'
    G1 = spdiags([-e 0*e e]/(2*dr), -1:1, Nsag, Nsag);
    G1(1,:) = 0;   G1(1,1:3) = [-3 4 -1]/(2*dr);
    G1(end,:) = 0; G1(end,end-2:end) = [1 -4 3]/(2*dr);
    d2f = G1*df;                                                        % f''
end


function [lmin, phi, ops] = lowest_eigenvalue(r, dw0, d2w0, N_r, dN_r, S0, nu, m)
% L_s = bending + prestress (N_r, dN_r already multiplied by S0) + FULL shell
% operator, clamped (r = alpha) / free (r = 1) BCs via a boundary projection.
% Returns the lowest REAL eigenvalue
% (no positivity filter; a negative value means the base state is unstable).
% Optional outputs: the eigenmode phi on the full grid (boundary values
% filled in by the projection) and the separate operators ops.Lb (bending),
% ops.Ls (in-plane prestress), ops.Lsh (shell), ops.P (projection).
    N = numel(r); dr = r(2) - r(1);
    N_theta = N_r + r.*dN_r + S0*r.^2;
    I = speye(N); D1 = sparse(N,N); D2 = D1; D3 = D1; D4 = D1;
    for i = 3:N-2
        D1(i,[i-1 i+1])         = [-1 1]/(2*dr);
        D2(i,[i-1 i i+1])       = [1 -2 1]/dr^2;
        D3(i,[i-2 i-1 i+1 i+2]) = [-1/2 1 -1 1/2]/dr^3;
        D4(i,i-2:i+2)           = [1 -4 6 -4 1]/dr^4;
    end
    Lb = sparse(N,N); Ls = sparse(N,N);
    for i = 3:N-2
        ri = r(i);
        Lb(i,:) = D4(i,:) + (2/ri)*D3(i,:) - ((2*m^2+1)/ri^2)*D2(i,:) ...
                + ((2*m^2+1)/ri^3)*D1(i,:) + ((m^4-4*m^2)/ri^4)*I(i,:);
        Ls(i,:) = -N_r(i)*D2(i,:) - (N_r(i)/ri + dN_r(i))*D1(i,:) ...
                + (m^2/ri^2)*N_theta(i)*I(i,:);
    end
    Lsh = sparse(build_shell_operator_full(r, dw0, d2w0, nu, m, 12*(1-nu^2)));
    Lsh([1 2 N-1 N],:) = 0;                 % only interior rows are used
    P   = boundary_projection(N, dr, nu, m);
    L   = Lb + Ls + Lsh;
    A   = full(L(3:N-2,:)*P);
    if nargout < 2
        ev  = eig(A);
        ev  = real(ev(abs(imag(ev)) < 1e-6*max(abs(ev))));
        lmin = min(ev);
    else
        [V, E] = eig(A); ev = diag(E);
        isreal_ev = abs(imag(ev)) < 1e-6*max(abs(ev));
        evr = real(ev); evr(~isreal_ev) = inf;
        [lmin, k] = min(evr);
        phi = real(P*V(:,k));
        [~, ipk] = max(abs(phi)); phi = phi/phi(ipk);   % normalise, peak = +1
        ops.Lb = Lb; ops.Ls = Ls; ops.Lsh = Lsh; ops.P = P;
    end
end


function P = boundary_projection(N, dr, nu, m)
% Clamped (r = alpha): w = w' = 0.  Free (r = 1): zero moment
%   w'' + nu (w' - m^2 w) = 0  and zero Kirchhoff shear
%   w''' + w'' - (1 + (2-nu) m^2) w' + (3-nu) m^2 w = 0,
% imposed with one-sided differences; P maps interior values to all nodes.
    A1 =  2/dr^2 + 1.5*nu/dr - nu*m^2;  A2 = -5/dr^2 - 2*nu/dr;
    A3 =  4/dr^2 + 0.5*nu/dr;           A4 = -1/dr^2;
    beta1 = 1 + (2-nu)*m^2;  beta2 = (3-nu)*m^2;
    B1 =  2.5/dr^3 + 2/dr^2 - 1.5*beta1/dr + beta2;
    B2 = -9/dr^3 - 5/dr^2 + 2*beta1/dr;
    B3 = 12/dr^3 + 4/dr^2 - 0.5*beta1/dr;
    B4 = -7/dr^3 - 1/dr^2;  B5 = 1.5/dr^3;
    Det = A1*B2 - A2*B1;
    P = sparse(N, N-4);
    for j = 1:N-4, P(j+2,j) = 1; end
    P(2,1) = 0.25;
    P(N-1,N-4) = (-A1*B3 + B1*A3)/Det;  P(N-1,N-5) = (-A1*B4 + B1*A4)/Det;  P(N-1,N-6) = (-A1*B5)/Det;
    P(N,  N-4) = (-B2*A3 + A2*B3)/Det;  P(N,  N-5) = (-B2*A4 + A2*B4)/Det;  P(N,  N-6) = ( A2*B5)/Det;
end

%% ========================================================================
%  FULL SHALLOW-SHELL (DMV) COUPLING OPERATOR
%  ========================================================================

function [S_shell, ops] = build_shell_operator_full(r, dw0, d2w0, nu, m, C_shell)
% BUILD_SHELL_OPERATOR_FULL  Exact linearised curvature coupling for an
% axisymmetric base state w0(r) and a mode  w' = phi(r) cos(m theta).
%
% Non-dimensional (r in R, w in h, in-plane displacement u in h^2/R,
% stress resultants N' in E h (h/R)^2). Perturbation membrane strains
%   e_rr = U' + w0' phi'
%   e_tt = U/r + m V/r
%   2e_rt = V' - V/r - m U/r - m w0' phi/r
% with u'_r = U cos(m th), u'_th = V sin(m th). Constitutive law
%   N_rr = (e_rr + nu e_tt)/(1-nu^2), N_tt = (e_tt + nu e_rr)/(1-nu^2),
%   N_rt = e_rt/(1+nu).
% In-plane equilibrium (no perturbation body force):
%   N_rr' + (m/r) N_rt + (N_rr - N_tt)/r = 0
%   N_rt' - (m/r) N_tt + 2 N_rt/r        = 0
% BCs: clamp (r = alpha): U = V = 0;  free edge (r = 1): N_rr = N_rt = 0.
% Transverse load:  N'_ij w0,ij = w0'' N_rr + (w0'/r) N_tt, so the term
% that enters L_s (same side as bending) is
%   S_shell*phi = -C_s*( w0'' N_rr + (w0'/r) N_tt ),   C_s = 12(1-nu^2).
% S_shell is quadratic in w0, even in w0 -> -w0, and positive
% semi-definite (= twice the perturbation stretching energy).
% Locally (constant curvature kappa, short waves) it reduces to
% C_s*kappa^2*phi (sphere); see the Supplementary Information.

    if nargin < 6, C_shell = 12*(1 - nu^2); end

    r = r(:); dw0 = dw0(:); d2w0 = d2w0(:);
    N = numel(r);
    dr = r(2) - r(1);

    Z = zeros(N);
    Rinv = diag(1./r);

    % first-derivative matrix on ALL nodes (2nd-order one-sided at ends)
    D1 = zeros(N);
    for i = 2:N-1
        D1(i,i-1) = -1/(2*dr);
        D1(i,i+1) =  1/(2*dr);
    end
    D1(1,1:3)   = [-3  4 -1]/(2*dr);
    D1(N,N-2:N) = [ 1 -4  3]/(2*dr);

    c = 1/(1 - nu^2);

    % strains: [operator on (U;V)] and [operator on phi]
    err_uv = [D1, Z];              err_p = diag(dw0)*D1;
    ett_uv = [Rinv, m*Rinv];       ett_p = Z;
    g_uv   = [-m*Rinv, D1 - Rinv]; g_p   = -m*diag(dw0./r);   % g = 2 e_rt

    Nrr_uv = c*(err_uv + nu*ett_uv);   Nrr_p = c*(err_p + nu*ett_p);
    Ntt_uv = c*(ett_uv + nu*err_uv);   Ntt_p = c*(ett_p + nu*err_p);
    Nrt_uv = g_uv/(2*(1 + nu));        Nrt_p = g_p/(2*(1 + nu));

    E1_uv = D1*Nrr_uv + m*Rinv*Nrt_uv + Rinv*(Nrr_uv - Ntt_uv);
    E1_p  = D1*Nrr_p  + m*Rinv*Nrt_p  + Rinv*(Nrr_p  - Ntt_p);
    E2_uv = D1*Nrt_uv - m*Rinv*Ntt_uv + 2*Rinv*Nrt_uv;
    E2_p  = D1*Nrt_p  - m*Rinv*Ntt_p  + 2*Rinv*Nrt_p;

    A = zeros(2*N, 2*N);
    B = zeros(2*N, N);
    k = 0;
    for i = 2:N-1
        k = k + 1; A(k,:) = E1_uv(i,:); B(k,:) = E1_p(i,:);
        k = k + 1; A(k,:) = E2_uv(i,:); B(k,:) = E2_p(i,:);
    end
    k = k + 1; A(k,1)   = 1;                              % U(alpha) = 0
    k = k + 1; A(k,N+1) = 1;                              % V(alpha) = 0
    k = k + 1; A(k,:) = Nrr_uv(N,:); B(k,:) = Nrr_p(N,:); % N_rr(1) = 0
    k = k + 1; A(k,:) = Nrt_uv(N,:); B(k,:) = Nrt_p(N,:); % N_rt(1) = 0

    UV  = -(A\B);            % (U;V) as a linear map of phi
    Nrr = Nrr_uv*UV + Nrr_p; % N'_rr as a linear map of phi
    Ntt = Ntt_uv*UV + Ntt_p;

    S_shell = -C_shell*(diag(d2w0)*Nrr + diag(dw0./r)*Ntt);

    if nargout > 1
        ops.UV = UV; ops.Nrr = Nrr; ops.Ntt = Ntt;
    end

end


function test_shell_operator_full(nu, m, alpha)
% Quick self-test of build_shell_operator_full:
%  (1) spherical cap, short-wave packet: (S phi)/(C_s kappa^2 phi) -> 1
%  (2) quadratic and even in w0
%  (3) positive Rayleigh quotient for clamped trial functions

    N = 301;
    r = linspace(alpha, 1, N).';
    Cs = 12*(1 - nu^2);

    kap = 50;
    S = build_shell_operator_full(r, kap*r, kap*ones(N,1), nu, m, Cs);
    r0 = 0.6; sig = 0.05; k = 150;
    phi = sin(k*(r - r0)).*exp(-((r - r0)/sig).^2);
    [~, j] = min(abs(r - r0 - pi/(2*k)));
    Sphi = S*phi;
    ratio1 = Sphi(j)/(Cs*kap^2*phi(j));

    dw0 = 60*(r - alpha); d2w0 = 60*ones(N,1);
    S1 = build_shell_operator_full(r, dw0, d2w0, nu, m, Cs);
    S2 = build_shell_operator_full(r, 2*dw0, 2*d2w0, nu, m, Cs);
    Sm = build_shell_operator_full(r, -dw0, -d2w0, nu, m, Cs);
    ratio2 = norm(S2)/norm(S1);
    even   = norm(Sm - S1)/norm(S1);

    wts = r*(r(2) - r(1));
    rng(0);                                  % reproducible self-test output
    rq = inf;
    for t = 1:50
        cc = randn(6,1);
        phi = zeros(N,1);
        for n = 1:6, phi = phi + cc(n)*(r - alpha).^(n+1); end
        rq = min(rq, (phi.'*(wts.*(S1*phi)))/(phi.'*(wts.*phi)));
    end

    fprintf('\nShell-operator self-test (m = %d):\n', m);
    fprintf('  sphere local limit ratio  = %.4f   (expect ~1)\n', ratio1);
    fprintf('  S(2 w0)/S(w0)             = %.4f   (expect 4)\n', ratio2);
    fprintf('  ||S(-w0)-S(w0)||/||S||    = %.1e  (expect 0)\n', even);
    fprintf('  min Rayleigh quotient     = %.3g   (expect > 0)\n\n', rq);

end


%% ========================================================================
%  SI FIG. S1: MODAL PROJECTIONS OF THE OPERATOR TERMS VS So
%  ========================================================================

function figS1 = make_fig_S1_operator_terms(X_list, alpha, nu, m, N, Nsag, npts)
% For each X = Gr/eps in X_list: find So_c, sweep So around it, and plot the
% modal (Rayleigh-quotient) projections of each operator term on the lowest
% eigenmode phi (weight r dr):
%   lambda = <phi,L_b phi> + <phi,L_s phi> + <phi,S phi>   (exact, sum checked)
% together with m^2 So. The whirling condition is where lambda = m^2 So.
    if nargin < 7, npts = 60; end
    figS1.X = X_list; figS1.data = cell(size(X_list));
    fsize = 20; lsize = 3;
    try
        clrs = distinguishable_colors(14);
    catch
        clrs = [0 0 1; 1 0 0; 0 0.8 0; 0 0 0.2; 0.5 0.5 0.5];
    end
    figure('Color','w','Position',[50 50 1500 450]);
    for ip = 1:numel(X_list)
        X = X_list(ip);
        Soc = critical_spin_cf(X, alpha, nu, m, N, Nsag);
        S0  = logspace(log10(max(5, Soc/20)), log10(20*Soc), npts);
        n = numel(S0);
        proj = zeros(n,4); lam = zeros(n,1); rpk = zeros(n,1); eres = zeros(n,1);
        for k = 1:n
            o = operator_terms_at_S0(X, S0(k), alpha, nu, m, N, Nsag);
            proj(k,:) = o.proj; lam(k) = o.lambda; rpk(k) = o.r_peak; eres(k) = o.eig_resid;
        end
        fprintf('\nGr/eps = %.1e:  So_c = %.4g\n', X, Soc);
        fprintf('  max |sum of projected terms| / lambda = %.2e\n', max(abs(sum(proj,2))./abs(lam)));
        fprintf('  max relative eigen-residual          = %.2e\n', max(eres));
        fprintf('  peak of mode ranges r = %.2f .. %.2f\n', min(rpk), max(rpk));

        % reference So^-2 line anchored to the shell term at the largest So
        % (the shell projection follows So^-2 above So_c and saturates below)
        sref = abs(proj(end,4))*(S0/S0(end)).^(-2);

        subplot(1, numel(X_list), ip); hold on
        plot(S0, abs(proj(:,1)), 'color', clrs(1,:), 'LineWidth', lsize, ...
             'DisplayName', '$|\partial_{\hat t\hat t}\hat w''|=\lambda$');
        plot(S0, abs(proj(:,2)), 'color', clrs(2,:), 'LineWidth', lsize, 'DisplayName', 'bending');
        plot(S0, abs(proj(:,3)), 'color', clrs(3,:), 'LineWidth', lsize, 'DisplayName', 'in-plane stress');
        plot(S0, abs(proj(:,4)), 'color', clrs(4,:), 'LineWidth', lsize, 'DisplayName', 'shell');
        plot(S0, m^2*S0, 'k:', 'LineWidth', lsize, 'DisplayName', '$m^2 So$');
        plot(S0, sref, ':', 'LineWidth', lsize, 'Color', [0.6 0.6 0.6], 'DisplayName', '$So^{-2}$');
        xline(Soc, '--', 'color', [0.6 0.6 0.6], 'LineWidth', lsize, 'DisplayName', '$So_c^{\rm w}$');
        set(gca,'XScale','log','YScale','log','FontSize',fsize,'TickLabelInterpreter','latex');
        xlim([S0(1) S0(end)]);
        ylim([1, 10^ceil(log10(max(abs(proj(:)))))]);
        xlabel('$So$','Interpreter','latex');
        if ip == 1, ylabel('Projected magnitude','Interpreter','latex'); end
        title(sprintf('$Gr/\\epsilon = 10^{%d}$', round(log10(X))), 'Interpreter','latex');
        if ip == 1
            try, legend('Interpreter','latex','Location','best','FontSize',12); catch, end
        end
        grid on; box on;

        figS1.data{ip} = struct('X', X, 'So_c', Soc, 'So', S0(:), 'proj', proj, ...
                                'lambda', lam, 'r_peak', rpk, 'eig_resid', eres);
    end
    assignin('base', 'figS1_data', figS1);
end


function o = operator_terms_at_S0(X, S0, alpha, nu, m, N, Nsag)
% Lowest eigenmode at (So, X) on the large-So base, and the projection of
% each operator term on it: proj = [-lambda, <B>, <stress>, <shell>] with
% <T> = sum_i r_i phi_i (T phi)_i / sum_i r_i phi_i^2 over interior nodes.
% sum(proj) = 0 to round-off (the projections add up to lambda).
    [rs, df, d2f] = sag_shape_cf(S0, alpha, nu, Nsag);
    r    = linspace(alpha, 1, N).';
    dw0  = (X/S0)*interp1(rs, df,  r, 'spline');
    d2w0 = (X/S0)*interp1(rs, d2f, r, 'spline');
    [C1, C2] = cf_constants(alpha, nu);
    Nr   = S0*(C1 + C2./r.^2 - (3+nu)/8*r.^2);
    dNr  = S0*(-2*C2./r.^3 - (3+nu)/4*r);
    [lam, phi, ops] = lowest_eigenvalue(r, dw0, d2w0, Nr, dNr, S0, nu, m);
    ii  = 3:N-2;  w = r(ii);
    den = sum(w.*phi(ii).^2);
    q   = @(T) sum(w.*phi(ii).*(T(ii,:)*phi))/den;
    o.proj = [-lam, q(ops.Lb), q(ops.Ls), q(ops.Lsh)];
    o.lambda = lam;
    Lphi = (ops.Lb(ii,:) + ops.Ls(ii,:) + ops.Lsh(ii,:))*phi;
    o.eig_resid = norm(Lphi - lam*phi(ii))/norm(lam*phi(ii));
    [~, ipk] = max(abs(phi)); o.r_peak = r(ipk);
    o.phi = phi; o.r = r;
end


function Soc = critical_spin_cf(X, alpha, nu, m, N, Nsag)
% So_c on the large-So base: scan downward, then fzero (as in the main sweep).
    res = @(S0) whirl_residual(S0, X, alpha, nu, m, N, Nsag);
    Sgrid = logspace(7.5, 0, 60); Soc = NaN;
    if res(Sgrid(1)) > 0, return; end
    for j = 2:numel(Sgrid)
        if res(Sgrid(j)) > 0
            Soc = fzero(res, [Sgrid(j) Sgrid(j-1)], optimset('TolX', 1e-6));
            return
        end
    end
end
