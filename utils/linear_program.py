import numpy as np
from scipy.optimize import linprog

class LinearProgram():
    def __init__(
        self,
        mode, # "minimize" or "maximize"
        obj,
        lhs_ineq,
        rhs_ineq,
        lhs_eq,
        rhs_eq,
        bounds,
        integrality
    ):
        assert(len(lhs_ineq) == len(rhs_ineq))
        assert(len(lhs_eq) == len(rhs_eq))
        
        self.mode = mode
        self.obj = obj
        if len(lhs_ineq) > 0:
            self.lhs_ineq = lhs_ineq
            self.rhs_ineq = rhs_ineq
        else:
            self.lhs_ineq = None
            self.rhs_ineq = None
            
        if len(lhs_eq) > 0:
            self.lhs_eq = lhs_eq
            self.rhs_eq = rhs_eq
        else:
            self.lhs_eq = None
            self.rhs_eq = None
            
        self.bounds = bounds
        self.integrality = integrality

        
    def __repr__(
        self,
    ):
        ret = ''
        ret += f'Mode: {self.mode}\n'
        ret += '\n'
        ret += f'Objective (Total {len(self.obj)} Coefs):\n{self.obj}\n'
        ret += '\n'

        if self.integrality == 0:
            int_str = 'Continuous variable, no integrality'
        elif self.integrality == 1:
            int_str = 'Integer variable'
            
        ret += f'Integrality: {self.integrality} ({int_str})\n'
        ret += '\n'
        ret += f'Constraints (Total {len(self.lhs_ineq)}):\n'
        for j in range(len(self.lhs_ineq)):
            ret += f'{self.lhs_ineq[j]} <= {self.rhs_ineq[j]}\n'

        if self.lhs_eq != None:
            for j in range(len(self.lhs_eq)):
                ret += f'{self.lhs_eq[j]} == {self.rhs_eq[j]}\n'
            
        ret += '\n'
        ret += f'Bounds (Total {len(self.bounds)}):\n'
        for b in self.bounds:
            ret += f'{b}'
        
        return ret

def create_primal(
    problem
):
    obj = problem.weight.flatten() # primal variable with the number of weights
    lhs_ineq = []
    rhs_ineq = []
    
    for i in range(problem.n_real_refugee): # Sum of assigned location for each refuge should be less or equal than 1
        coef = np.zeros(obj.shape)
        for l in range(problem.n_location):
            coef[i * problem.n_location + l] = 1
        lhs_ineq.append(coef)
        rhs_ineq.append(1)
        
    lhs_eq = []
    rhs_eq = []
    for i in range(problem.n_real_refugee, problem.n_refugee): # Sum of assigned location for each refuge should be less or equal than 1
        coef = np.zeros(obj.shape)
        for l in range(problem.n_location):
            coef[i * problem.n_location + l] = 1
        lhs_eq.append(coef)
        rhs_eq.append(1)

    for l in range(problem.n_location): # Sum of assigned refugee for each location should be less or equal than capacity
        coef = np.zeros(obj.shape)
        for i in range(problem.n_refugee):
            coef[i * problem.n_location + l] = 1
        lhs_ineq.append(coef)
    rhs_ineq.extend(problem.capacity)

    bounds = []
#     for j in range(len(obj)): # Our primal variables are bounded in [0, 1]
#         bounds.append((0, 1))
    bounds = (0, 1)

    lhs_ineq = np.array(lhs_ineq)
    rhs_ineq = np.array(rhs_ineq)
    
    lhs_eq = np.array(lhs_eq)
    rhs_eq = np.array(rhs_eq)
        
    primal = LinearProgram(
        mode="maximize",
        obj=obj,
        lhs_ineq=lhs_ineq,
        rhs_ineq=rhs_ineq,
        lhs_eq=lhs_eq,
        rhs_eq=rhs_eq,
        bounds=bounds,
        integrality=1,
    )

    return primal

def create_dual(
    problem
):
    # First n_refugee variables would be u_i, and next n_location variables would be v_l
    coef_u = np.ones(problem.n_refugee)
    coef_v = problem.capacity.astype(float)
    obj = np.concatenate((coef_u, coef_v))
    
    lhs_ineq = []
    rhs_ineq = []

    for i in range(problem.n_refugee):
        for l in range(problem.n_location):
            coef = np.zeros(problem.n_refugee + problem.n_location)
            coef[i] = 1
            coef[problem.n_refugee + l] = 1

            # Multiply -1 to change direction of inequality
            lhs_ineq.append(-1 * coef)
            rhs_ineq.append(-1 * problem.weight[i][l])

#     bounds = []

#     real_refugee_bounds = [(0, None) for i in problem.real_refugees]
#     dummy_bounds = [(None, None) for i in problem.dummies]
#     location_bounds = [(0, None) for l in problem.locations]
    
#     for j in range(len(obj)):
#         bounds.append((0, None))
    bounds = (0, None)

    dual = LinearProgram(
        mode="minimize",
        obj=obj,
        lhs_ineq=lhs_ineq,
        rhs_ineq=rhs_ineq,
        lhs_eq=[],
        rhs_eq=[],
        bounds=bounds,
        integrality=0,
    )

    return dual

def solve_lp(
    lp,
):
    if lp.mode=='maximize':
        obj = -1 * lp.obj
    elif lp.mode=='minimize':
        obj = lp.obj
    else:
        assert(0)
        
    opt = linprog(
        c=obj,
        A_ub=lp.lhs_ineq,
        b_ub=lp.rhs_ineq,
        A_eq=lp.lhs_eq,
        b_eq=lp.rhs_eq,
        bounds=lp.bounds,
        integrality=lp.integrality,
        method='highs',
    )
    
    if opt.status != 0:
        print(opt.message)
        assert(0)

    sol = opt.x
    val = opt.fun

    if lp.mode=='maximize':
        val = -1 * val
    
    return sol, val