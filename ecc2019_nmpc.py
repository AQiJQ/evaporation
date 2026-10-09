"""Isolated ECC2019 Eq.(3)/(8) reproduction with documented assumptions.

Exploratory reproduction, not used as the primary quantitative benchmark.

This is NOT an exact authors' reproduction or a proved stabilizing controller.
No existing SAC/safety model is changed. Optional CasADi/SciPy live in a separate
environment. Only explicit train-ecc invokes ECC parameter fitting; never SAC.
"""
from dataclasses import dataclass, asdict
import numpy as np

LABEL = 'ECC2019_reproduction_with_documented_assumptions'


@dataclass(frozen=True)
class ECCSettings:
    horizon: int = 10                 # specified_in_paper
    damping_alpha: float = .01        # specified_in_paper; NOT SAC entropy
    policy_update_steps: int = 500    # specified_in_paper
    exploration_probability: float = .1
    gamma: float = .99                # not specified in the paper
    pd_floor: float = 1e-6            # implementation choice for reproduction
    solver_scale: float = 10000.      # numerical objective scaling only
    solver_tol: float = 1e-8
    solver_max_iter: int = 300
    fit_window: int = 32              # n in Eq.(8) not numerically specified
    fit_every: int = 1                # full fit each step, as Section IV
    fit_max_iter: int = 100
    fit_tol: float = 1e-8
    quadratic_coordinates: str = 'physical_deviation_from_nominal_steady'  # not specified


def _symmetric(values, n):
    h = np.zeros((n,n)); at = 0
    for i in range(n):
        for j in range(i,n): h[i,j] = h[j,i] = values[at]; at += 1
    return h


def blocks(theta):
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (33,) or not np.isfinite(theta).all(): raise ValueError('Expected finite 33D ECC theta')
    return {'H_lambda':_symmetric(theta[:3],2),'h_lambda':theta[3:5],'c_lambda':theta[5],
            'H_terminal':_symmetric(theta[6:9],2),'h_terminal':theta[9:11],'c_terminal':theta[11],
            'H_stage':_symmetric(theta[12:22],4),'h_stage':theta[22:26],'c_stage':theta[26],
            'model_bias':theta[27:29],'state_lower':theta[29:31],'state_upper':theta[31:33]}


def initial_theta(settings=ECCSettings()):
    p=np.zeros(33); p[12:22]=np.eye(4)[np.triu_indices(4)]
    # Paper says Vf=0 and also insists Vf positive definite: explicit regularization.
    p[6:9]=settings.pd_floor*np.eye(2)[np.triu_indices(2)]
    p[29:33]=[25,40,100,80]
    return p


def theta_certificate(theta, settings=ECCSettings()):
    b=blocks(theta)
    margins=np.r_[np.linalg.eigvalsh(b['H_stage'])-settings.pd_floor,
                  np.linalg.eigvalsh(b['H_terminal'])-settings.pd_floor,
                  b['state_upper']-b['state_lower']-1e-6]
    return {'passed':bool(np.min(margins)>=-1e-10),'margins':margins.tolist(),
            'stage_min_eigenvalue':float(np.linalg.eigvalsh(b['H_stage']).min()),
            'terminal_min_eigenvalue':float(np.linalg.eigvalsh(b['H_terminal']).min()),
            'nominal_closed_loop_stability_proved':False}


def symbolic_step(ca, cfg, x, u):
    """Same existing nominal nonlinear RK4, no parameterized plant alteration."""
    def derivative(x):
        f1,x1,t1,t200=cfg.disturbance_nominal
        t2=cfg.t2_p_coeff*x[1]+cfg.t2_x_coeff*x[0]+cfg.t2_offset
        t3=cfg.t3_p_coeff*x[1]+cfg.t3_offset
        t100=cfg.t100_p_coeff*u[0]+cfg.t100_offset
        q100=cfg.ua1_factor*(f1+cfg.recirculation_f3)*(t100-t2)
        f4=(q100-f1*cfg.cp*(t2-t1))/cfg.latent_evaporation
        q200=cfg.ua2*(t3-t200)/(1+cfg.ua2/(2*cfg.cp*ca.fmax(u[1],1e-6)))
        return ca.vertcat((f1*x1-(f1-f4)*x[0])/cfg.liquid_holdup,
                          (f4-q200/cfg.latent_evaporation)/cfg.pressure_capacitance)
    h=cfg.dt_min; k1=derivative(x);k2=derivative(x+h*k1/2);k3=derivative(x+h*k2/2);k4=derivative(x+h*k3)
    return x+h*(k1+2*k2+2*k3+k4)/6


class ECCNMPC:
    """86-variable non-condensed NLP with soft state bounds and hard inputs."""
    def __init__(self, model, settings=ECCSettings()):
        try: import casadi as ca
        except ImportError as exc: raise RuntimeError('ECC solver requires casadi in the isolated ECC environment') from exc
        self.ca,self.model,self.cfg,self.settings=ca,model,model.cfg,settings
        self.theta=initial_theta(settings);self.warm={};self.calls=0
        for fixed in (False,True): self._build(fixed)

    def _build(self, fixed):
        ca,c,s=self.ca,self.cfg,self.settings; n=s.horizon
        if n!=10: raise ValueError('Main ECC comparison fixes paper N=10')
        theta=ca.MX.sym('theta',33);state=ca.MX.sym('state',2);first=ca.MX.sym('first',2)
        x=ca.MX.sym('x',2,n+1);u=ca.MX.sym('u',2,n);slack=ca.MX.sym('slack',4,n+1)
        if s.quadratic_coordinates!='physical_deviation_from_nominal_steady':
            raise ValueError('Only the predeclared steady-deviation quadratic basis is supported')
        x_origin=np.array([25.,49.743]);u_origin=self.model.steady_input(x_origin)
        def sym(values,d):
            h=ca.MX.zeros(d,d);at=0
            for i in range(d):
                for j in range(i,d): h[i,j]=h[j,i]=values[at];at+=1
            return h
        def quad(v,H,h,k): return ca.mtimes([v.T,H,v])/2+ca.dot(h,v)+k
        Hl=sym(theta[12:22],4);Ht=sym(theta[6:9],2);Ha=sym(theta[:3],2)
        objective=quad(x[:,0]-x_origin,Ha,theta[3:5],theta[5])+s.gamma**n*quad(x[:,n]-x_origin,Ht,theta[9:11],theta[11])
        g=[x[:,0]-state];lbg=[0,0];ubg=[0,0]
        if fixed:g.append(u[:,0]-first);lbg += [0,0];ubg += [0,0]
        for k in range(n+1):
            objective += s.gamma**k*(ca.dot(slack[:,k],slack[:,k])+ca.sum1(slack[:,k]))
            # Printed paper sign is inconsistent with h<=slack: document correction.
            g.append(ca.vertcat(theta[29:31]-x[:,k],x[:,k]-theta[31:33])-slack[:,k])
            lbg += [-np.inf]*4;ubg += [0]*4
            if k<n:
                objective += s.gamma**k*quad(ca.vertcat(x[:,k]-x_origin,u[:,k]-u_origin),Hl,theta[22:26],theta[26])
                g.append(x[:,k+1]-symbolic_step(ca,c,x[:,k],u[:,k])-theta[27:29])
                lbg += [0,0];ubg += [0,0]
        decision=ca.vertcat(ca.vec(x),ca.vec(u),ca.vec(slack));parameters=ca.vertcat(theta,state,first);constraints=ca.vertcat(*g)
        lbd=np.r_[np.full(2*(n+1),-np.inf),np.tile(c.input_lower,n),np.zeros(4*(n+1))]
        ubd=np.r_[np.full(2*(n+1),np.inf),np.tile(c.input_upper,n),np.full(4*(n+1),np.inf)]
        name='ecc_Q' if fixed else 'ecc_V'
        solver=ca.nlpsol(name,'ipopt',{'x':decision,'p':parameters,'f':objective/s.solver_scale,'g':constraints},
                        {'print_time':False,'ipopt.print_level':0,'ipopt.sb':'yes','ipopt.tol':s.solver_tol,
                         'ipopt.max_iter':s.solver_max_iter,'ipopt.bound_relax_factor':0.})
        multiplier=ca.MX.sym('lam',constraints.numel())
        lagrange=objective/s.solver_scale+ca.dot(multiplier,constraints)
        gradient=ca.Function(name+'_gradient',[decision,parameters,multiplier],[s.solver_scale*ca.gradient(lagrange,theta)])
        check=ca.Function(name+'_constraints',[decision,parameters],[constraints])
        setattr(self,'Q' if fixed else 'V',(solver,gradient,check,lbd,ubd,np.array(lbg),np.array(ubg)))

    def solve(self, state, theta=None, first_action=None, gradient=False):
        theta=self.theta if theta is None else np.asarray(theta)
        if not theta_certificate(theta,self.settings)['passed']:raise ValueError('ECC stage/terminal PD gate failed')
        fixed=first_action is not None;solver,grad,check,lbd,ubd,lbg,ubg=self.Q if fixed else self.V
        n=self.settings.horizon;x=np.asarray(state,dtype=float);b=blocks(theta)
        first=np.zeros(2) if not fixed else np.asarray(first_action,dtype=float)
        if fixed and (np.any(first<self.cfg.input_lower) or np.any(first>self.cfg.input_upper)):raise ValueError('Q first input outside hard bounds')
        # Use feasible rollout initialization rather than invent an online fallback.
        us=np.tile(np.clip(self.model.steady_input(self.cfg.robust_economic_reference_state),self.cfg.input_lower,self.cfg.input_upper),(n,1))
        if fixed:us[0]=first
        xs=[x]
        for k in range(n):xs.append(self.model.step(xs[-1],us[k],self.cfg.disturbance_nominal)+b['model_bias'])
        xs=np.asarray(xs);slacks=np.maximum(np.c_[b['state_lower']-xs,xs-b['state_upper']],0)+1e-5
        guess=np.r_[xs.ravel(),us.ravel(),slacks.ravel()]
        if fixed in self.warm:guess=self.warm[fixed].copy()
        p=np.r_[theta,x,first]
        result=solver(x0=guess,p=p,lbx=lbd,ubx=ubd,lbg=lbg,ubg=ubg)
        self.calls+=1;z=np.array(result['x']).ravel();gv=np.array(check(z,p)).ravel()
        violation=max(float(np.maximum(lbg-gv,0).max()),float(np.maximum(gv-ubg,0).max()),
                      float(np.maximum(lbd-z,0).max()),float(np.maximum(z-ubd,0).max()))
        if not solver.stats()['success'] or not np.isfinite(z).all() or violation>1e-5:
            raise RuntimeError(f'ECC NLP failed: {solver.stats()["return_status"]}, violation={violation}; no silent backup')
        self.warm[fixed]=z.copy();xs=z[:2*(n+1)].reshape(n+1,2);us=z[2*(n+1):2*(n+1)+2*n].reshape(n,2)
        slack=z[-4*(n+1):].reshape(n+1,4)
        canonical=np.maximum(np.c_[b['state_lower']-xs,xs-b['state_upper']],0)
        return {'value':float(result['f'])*self.settings.solver_scale,'action':us[0].copy(),
                'states':xs,'controls':us,'slack':slack,'first_slack':canonical[0].copy(),
                'raw_first_slack':slack[0].copy(),'minimum_required_slack':canonical,
                'solver_status':solver.stats()['return_status'],'constraint_residual':violation,
                'gradient':np.array(grad(z,p,result['lam_g'])).ravel() if gradient else None}


def fit_td(solver, transitions, theta, target_theta):
    """Eq.(8) full nonlinear least-squares fit with PD gates, not SAC updates."""
    from scipy.optimize import minimize
    targets=np.array([ell+solver.settings.gamma*solver.solve(xn,target_theta)['value'] for x,u,xn,ell in transitions])
    cache={}
    def objective(candidate):
        key=np.asarray(candidate).tobytes()
        if key in cache:return cache[key]
        values=[solver.solve(x,candidate,first_action=u,gradient=True) for x,u,xn,ell in transitions]
        residual=np.array([v['value'] for v in values])-targets
        grads=np.array([v['gradient'] for v in values]);scale=200.**2
        output=(float(np.mean(residual**2)/scale),2*residual@grads/len(values)/scale)
        cache.clear();cache[key]=output;return output
    # Fixed c_lambda=0 is a documented value-offset gauge; no policy effect at a given state.
    bounds=[(None,None)]*33;bounds[5]=(0.,0.)
    # SLSQP may evaluate invalid trial theta despite inequalities: forbid silently
    # evaluating indefinite costs by a smooth feasibility extension outside the gate.
    def safe_objective(candidate):
        certificate=theta_certificate(candidate,solver.settings)
        deficit=np.maximum(-np.array(certificate['margins']),0)
        if not certificate['passed']:
            penalty=1e10+1e12*float(deficit@deficit)
            return penalty,np.zeros(33)
        return objective(candidate)
    before=float(objective(theta)[0])
    fit=minimize(safe_objective,theta,method='SLSQP',jac=True,bounds=bounds,
                 constraints=[{'type':'ineq','fun':lambda p:np.array(theta_certificate(p,solver.settings)['margins'])}],
                 options={'maxiter':solver.settings.fit_max_iter,'ftol':solver.settings.fit_tol,'disp':False})
    if not fit.success or not theta_certificate(fit.x,solver.settings)['passed']:
        raise RuntimeError(f'Eq8 full-convergence fit failed: {fit.message}; do not label this trained ECC2019')
    if fit.fun>before+1e-8*max(1.,before):
        raise RuntimeError('Eq8 fit increased TD objective; reject, do not silently deploy')
    fitted=theta+solver.settings.damping_alpha*(fit.x-theta)
    if not theta_certificate(fitted,solver.settings)['passed']:raise RuntimeError('Damped ECC theta PD gate failed')
    return fitted,{'TD_objective_before':before,'TD_objective_after_full_fit':float(fit.fun),
                   'fit_iterations':int(fit.nit),'fit_converged':True,'transition_count':len(transitions)}
