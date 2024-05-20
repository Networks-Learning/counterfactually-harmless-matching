from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
from copy import deepcopy
from collections import defaultdict
from .linear_program import create_primal, create_dual, solve_lp
from .common import reset_seed, LOCATIONS

class Problem():
    def __init__(
        self,
        real_refugees=[],
        n_real_refugee=0, # ignored if refugees != None
        dummies=[],
        locations=[],
        n_location=0, # ignored if locations != None
        capacity=[],
        weight=[],
    ):
        assert(real_refugees!=None or n_real_refugee!=None)
        assert(locations!=None or n_location!=None)

        if len(real_refugees) == 0:
            self.n_real_refugee = n_real_refugee
            self.real_refugees = [f"i_{i}" for i in range(self.n_real_refugee)]
        else:
            self.real_refugees = real_refugees
            self.n_real_refugee = len(real_refugees)

        if len(locations) == 0:
            self.n_location = n_location
            self.locations = [f"l_{l}" for l in range(self.n_location)]
        else:
            self.locations = locations
            self.n_location = len(locations)

        if len(capacity) == 0:
            assert(0)
        else:
            self.capacity = capacity

        if len(weight) == 0:
            assert(0)
        else:
            self.weight = weight

        if len(dummies) == 0:
            self.n_dummy = np.sum(self.capacity) - self.n_real_refugee
            self.dummies = [f"dummy_{i}" for i in range(self.n_dummy)]
    
            self.refugees = self.real_refugees + self.dummies
            self.n_refugee = self.n_real_refugee + self.n_dummy
            
            if self.n_dummy > 0:
                self.dummy_weight = np.ones((self.n_dummy, self.n_location)) * 0
                self.weight = np.concatenate((self.weight, self.dummy_weight), axis=0)
        else:
            self.n_dummy = len(dummies)
            self.dummies = dummies

            self.refugees = self.real_refugees + self.dummies
            self.n_refugee = self.n_real_refugee + self.n_dummy

    def __repr__(
        self
    ):
        weight_df = pd.DataFrame(self.weight, columns=self.locations, index=self.refugees)
        df_str = weight_df.applymap(lambda x: '{0:.4f}'.format(x))
        
        ret = ''
        ret += f"Number of Refugees: {self.n_refugee}\n"
        ret += f"Number of Locations: {self.n_location}\n"
        ret += f"Location Capacity: {self.capacity}\n"
        ret += '\n'
        ret += f"Classifier Weight:\n{df_str}"
        
        return ret

def sol2match(
    problem,
    sol
):
    assert(set(sol).issubset({0, 1}))

    sol = sol.reshape(problem.weight.shape)
    # assert(set(np.sum(sol, axis=1)) == {1})

    match_index = np.argmax(sol, axis=1)

    match = []
    for i, loc in enumerate(match_index):
        pair = (problem.refugees[i], problem.locations[loc])
        match.append(pair)

    return match

def match2sol(
    problem,
    match
):
    assert(len(match) == problem.n_refugee)
    sol = np.zeros(problem.weight.shape)
    for edge in match:
        ref_index = problem.refugees.index(edge[0])
        loc_index = problem.locations.index(edge[1])

        sol[ref_index][loc_index] = 1

    return sol.flatten()
    
def maximum_assign(scores, capacity):
    n_refugee, n_location = scores.shape
    problem = Problem(
        n_real_refugee=n_refugee,
        n_location=n_location,
        capacity=capacity,
        weight=scores,
    )
    
    primal = create_primal(problem)
    primal_sol, primal_val = solve_lp(primal)
    primal_match = sol2match(problem, primal_sol)
    
    assignment_list = [
        (int(i.split('_')[1]), int(l.split('_')[1]))
        for i, l in primal_match
    ]
    assignment_list.sort(key=lambda x: x[0])
    assignment_list = [l for i, l in assignment_list]

    return assignment_list

def make_assignments(
    location_probs,
    capacity_df,
    refugee_batch_size=100,
    refugee_batch_num=5000,
    location_num=10,
    policy='maximum',
    seed=0
):
    reset_seed(seed)
    location_probs = deepcopy(location_probs)
    capacity_df = deepcopy(capacity_df)
    
    location_probs = location_probs.reshape(refugee_batch_num, refugee_batch_size, location_num)

    assignments = []
    for i, batch_scores in enumerate(tqdm(location_probs, desc='Assigning Refugees')):
        batch_capacity = capacity_df.iloc[i].values

        if policy == 'maximum':
            batch_locations = maximum_assign(batch_scores, batch_capacity)
        else:
            assert(0)

        assignments.extend(batch_locations)
        
    assignments = np.array(assignments).reshape(refugee_batch_num, refugee_batch_size)

    return assignments
    
def get_batch_problem(
    data_df,
    capacity_df,
):
    def get_batch_num(x):
        return int(x.split('.')[1])
    
    assigned = data_df['refugee_id'].apply(get_batch_num)
    batches = sorted(list(set(assigned)))
    
    problems = []
    for b in tqdm(batches, desc='Creating Problems'):
        cur_batch = data_df.loc[assigned == b]
        cur_weight = cur_batch.drop(columns=['refugee_id']).to_numpy()
        cur_capacity = capacity_df.loc[int(b)]
        
        problem = Problem(
            real_refugees=cur_batch['refugee_id'].tolist(),
            locations=cur_capacity.index.tolist(),
            capacity=cur_capacity.to_numpy(),
            weight=cur_weight
        )
        
        problems.append(problem)
        
    return problems
    
def get_success_edges(
    ref_id,
    assignment,
    employments
):
    refugee_batch_num, refugee_batch_size = assignment.shape
    employments = employments.reshape(
        refugee_batch_num,
        refugee_batch_size,
        -1
    )
    ref_id = ref_id.reshape(refugee_batch_num, refugee_batch_size)
    
    success_edges = []
    for i, (ref, ass, emp) in enumerate(zip(ref_id, assignment, employments)):
        se = [(ref[j], LOCATIONS[a]) for j, a in enumerate(ass) if emp[j][a]]
        success_edges.append(se)
        
    return success_edges
    
def load_problems(
    problem_save_path
):
    with open(problem_save_path / 'original_problems', 'rb') as f:
        problems = pickle.load(f)
    with open(problem_save_path / 'success_edges', 'rb') as f:
        success_edges = pickle.load(f)
    with open(problem_save_path / 'new_problems', 'rb') as f:
        new_problems = pickle.load(f)
            
    return problems, success_edges, new_problems

def get_adjusted_weights(
    problem_save_path,
    problems,
    success_edges,
    epsilon=1e-4,
):    
    new_problems = []
    for problem, success_edge in zip(tqdm(problems, desc='Solving Problems'), success_edges):
        dual = create_dual(problem)
        dual_sol, dual_val = solve_lp(dual)

        if len(success_edge) == len(problem.refugees):
            reduced_problem = None
            reduced_primal_sol = None
        else:
            reduced_problem = reduce_problem(problem, success_edge)
            reduced_primal = create_primal(reduced_problem)

            reduced_primal_sol, reduced_primal_val = solve_lp(reduced_primal)

        sol, harmless_edge, reduced_optimal_edge, location_counter = aggregate_solution(
            problem,
            reduced_problem,
            success_edge,
            reduced_primal_sol
        )

        g_prime = build_inverse_solution(
            problem,
            dual_sol,
            harmless_edge,
        )

        g_breve = modify_inverse_solution(
            problem,
            g_prime,
            success_edge,
            epsilon=epsilon,
        )

        new_problem = Problem(
            real_refugees=problem.real_refugees,
            dummies=problem.dummies,
            locations=problem.locations,
            capacity=problem.capacity,
            weight=g_breve,
        )
            
        sol_star = match2sol(new_problem, harmless_edge)
        
        value_ret = optimality_value_test(new_problem, sol_star)
        if not value_ret:
            assert(0)
        inclusion_ret = optimality_inclusion_test(new_problem, sol_star, success_edge)
        if not inclusion_ret:
            assert(0)
            
        new_problems.append(new_problem)
            
    problem_save_path.mkdir(exist_ok=True, parents=True)
    with open(problem_save_path / 'original_problems', 'wb') as f:
        pickle.dump(problems, f, pickle.HIGHEST_PROTOCOL)
    with open(problem_save_path / 'success_edges', 'wb') as f:
        pickle.dump(success_edges, f, pickle.HIGHEST_PROTOCOL)
    with open(problem_save_path / 'new_problems', 'wb') as f:
        pickle.dump(new_problems, f, pickle.HIGHEST_PROTOCOL)
            
    return new_problems

def reduce_problem(
    problem,
    success_edge,
):
    reduced_refugees = deepcopy(problem.refugees)
    reduced_locations = deepcopy(problem.locations)
    
    reduced_capacity = problem.capacity.tolist()
    reduced_weight = deepcopy(problem.weight)

    for se in success_edge:
        reduced_refugees.remove(se[0])
        loc_index = reduced_locations.index(se[1])
        reduced_capacity[loc_index] -= 1
        if reduced_capacity[loc_index] == 0:
            reduced_capacity.pop(loc_index)
            reduced_locations.pop(loc_index)

    reduced_capacity = np.array(reduced_capacity)
    temp_i = [problem.refugees.index(i) for i in reduced_refugees]
    temp_l = [problem.locations.index(l) for l in reduced_locations]

    reduced_weight = reduced_weight.take(temp_i, axis=0).take(temp_l, axis=1)

    reduced_dummies = [ref for ref in reduced_refugees if ref in problem.dummies]
    reduced_real_refugees = [ref for ref in reduced_refugees if ref in problem.real_refugees]
    
    reduced_problem = Problem(
        real_refugees=reduced_real_refugees,
        dummies=reduced_dummies,
        locations=reduced_locations,
        capacity=reduced_capacity,
        weight=reduced_weight,
    )

    return reduced_problem

def aggregate_solution(
    problem,
    reduced_problem,
    success_edge,
    reduced_primal_solution,
):
    if reduced_problem == None:
        reduced_optimal_edge = []
    else:
        reduced_optimal_edge = sol2match(reduced_problem, reduced_primal_solution)
    harmless_edge = success_edge + reduced_optimal_edge

    assert(len(set([edge[0] for edge in harmless_edge])) == problem.n_refugee)

    harmless_edge = sorted(harmless_edge, key=lambda m: m[0])
    
    # Check if every refugee is assigned to one location and capacity
    location_counter = defaultdict(int)
    for edge in harmless_edge:
        location_counter[edge[1]] += 1

    for k, v in location_counter.items():
        location_index = problem.locations.index(k)
        assert(v <= problem.capacity[location_index])

    sol = match2sol(problem, harmless_edge)
    
    return sol, harmless_edge, reduced_optimal_edge, location_counter

def build_inverse_solution(
    problem,
    dual_sol,
    harmless_edge,
):
    new_weight = deepcopy(problem.weight)

    u_star = dual_sol[:problem.n_refugee]
    v_star = dual_sol[problem.n_refugee:]

    for m in harmless_edge:
        ref_index = problem.refugees.index(m[0])
        loc_index = problem.locations.index(m[1])

        new_weight[ref_index][loc_index] = u_star[ref_index] + v_star[loc_index]

    return new_weight

def modify_inverse_solution(
    problem,
    g_prime,
    success_edge,
    epsilon=1e-4,
):
    new_weight = deepcopy(g_prime)
    for edge in success_edge:
        ref_index = problem.refugees.index(edge[0])
        loc_index = problem.locations.index(edge[1])
        
        new_weight[ref_index][loc_index] += epsilon
        
    return new_weight

def optimality_value_test(
    problem,
    target_sol
):
    primal = create_primal(problem)
    primal_sol, primal_val = solve_lp(primal)

    w = problem.weight.flatten()
    target_val = np.dot(w, target_sol)

    value_ret = (abs(target_val - primal_val) < 1e-6)

    return value_ret, target_val, primal_val

def optimality_inclusion_test(
    problem,
    target_sol,
    m_prime
):
    primal = create_primal(problem)
    primal_sol, primal_val = solve_lp(primal)

    primal_match = sol2match(problem, primal_sol)
    
    prime_ret = set(m_prime).issubset(set(primal_match))

    return prime_ret