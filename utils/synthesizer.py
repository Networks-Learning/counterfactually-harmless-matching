from .common import LOCATIONS, reset_seed, calculate_beta_distribution_parameter
from .problem import Problem, sol2match
from .linear_program import create_primal, solve_lp

import re
from collections import defaultdict, Counter
from pathlib import Path
import pandas as pd
import random
from tqdm import tqdm
import numpy as np
import copy

from qpsolvers import solve_ls

FIELDS = [
    'origin', 'age', 'education', 'gender',
]

AGE_GROUPS = ['20-24', '25-34', '35-44', '45-64', '65-100']
EDUCATION_GROUPS = ['primary or less', 'secondary', 'tertiary']
GENDER_GROUPS = ['Male', 'Female']
CONTINENT_GROUPS = ['Africa', 'Asia', 'Europe', 'Latin America', 'Northern America', 'Oceania']

AFRICA = [
    'Algeria', 'Angola', 'Benin' 'Burkina Faso', 'Burundi',
    'Cabo Verde', 'Cameroon', 'Central African Republic', 'Chad', 'Congo',
    'Cote d\'Ivoire', 'Djibouti', 'Egypt', 'Equatorial Guinea', 'Eritrea',
    'Ethiopia', 'Gabon', 'Gambia','Ghana', 'Guinea',
    'Guinea-Bissau', 'Kenya', 'Liberia', 'Libya', 'Madagascar',
    'Malawi', 'Mali', 'Mauritania', 'Morocco', 'Nigeria',
    'Rwanda', 'Senegal', 'Sierra Leone', 'Somalia', 'South Africa', 
    'South Sudan', 'Sudan', 'Tanzania', 'Togo', 'Tunisia',
    'Uganda', 'Zambia', 'Zimbabwe',
    
]

ASIA = [
    'Afghanistan', 'Armenia', 'Azerbaijan', 'Bangladesh', 'Bhutan',
    'Cambodia', 'China', 'Georgia', 'Hong Kong', 'India',
    'Indonesia', 'Iran', 'Iraq', 'Israel', 'Japan',
    'Jordan', 'Kazakhstan', 'Kuwait', 'Kyrgyzstan', 'Laos',
    'Lebanon', 'Malaysia', 'Mongolia', 'Myanmar', 'Nepal',
    'North Korea', 'Pakistan', 'Palestinian', 'Philippines', 'Qatar',
    'Russia', 'Saudi Arabia', 'Singapore', 'South Korea', 'Sri Lanka',
    'Syria', 'Tajikistan', 'Thailand', 'Turkey', 'Turkmenistan',
    'United Arab Emirates', 'Uzbekistan', 'Vietnam', 'Yemen', 
]

EUROPE = [
    'Albania', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Cyprus', 
    'France', 'Germany', 'Greece', 'Hungary', 'Italy',
    'Latvia', 'Lithuania', 'Macedonia', 'Moldova', 'Montenegro', 
    'Netherlands', 'Poland', 'Portugal', 'Romania', 'Serbia',
    'Slovakia', 'Slovenia', 'Spain', 'Ukraine', 'United Kingdom',
    
]

LATIN_AMERICA = [
    'Antigua and Barbuda', 'Argentina', 'Bahamas', 'Barbados', 'Belize',
    'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Costa Rica',
    'Cuba', 'Dominica', 'Ecuador', 'El Salvador', 'Guadeloupe',
    'Guatemala', 'Guyana', 'Haiti', 'Honduras', 'Jamaica', 'Mexico', 
    'Nicaragua', 'Panama', 'Paraguay', 'Peru', 'St. Lucia',
    'St. Vincent and the Grenadines', 'Suriname', 'Trinidad and Tobago', 'Turks and Caicos Islands', 'Uruguay',
    'Venezuela', 
]

NORTHERN_AMERICA = [
    'Canada',
]

OCEANIA = [
    'Fiji', 'Vanuatu',
]

def get_continent(country):
    continent = None
    for continent in [AFRICA, ASIA, EUROPE, LATIN_AMERICA, NORTHERN_AMERICA, OCEANIA]:
        if country in continent:
            break
            
    if continent == AFRICA:
        return 'Africa'
    elif continent == ASIA:
        return 'Asia'
    elif continent == EUROPE:
        return 'Europe'
    elif continent == LATIN_AMERICA:
        return 'Latin America'
    elif continent == NORTHERN_AMERICA:
        return 'Northern America'
    elif continent == OCEANIA:
        return 'Oceania'
    else:
        assert(0)

def load_refugee_distribution(
    data_path = './data/statistics/refined_refugee_demographics.csv'
):
    df = pd.read_csv(data_path)

    dist_dict = defaultdict(int)
    loc_dict = defaultdict(int)
    for i, row in df.iterrows():
        country = row['Country of origin']
        country = re.sub('\s*\([a-zA-z\s\.]*\)s*', '', country).strip()
        
        if country == 'Unknown':
            continue

        location = row['Location Description']
        
        row = row.drop(index = ['Country of origin', 'Location Description'])
        for item in row.items():
            gender_age, val = item
            age = ' '.join(gender_age.split()[1:])
            gender = ' '.join(gender_age.split()[0])
            gender = gender.replace(' ', '')

            if age != '18 - 59':
                continue
            
            case = f'{country}_{gender}'
            dist_dict[case] += val
            loc_dict[location] += val
    
    dist_df = []
    for key, val in dist_dict.items():
        dist_df.append([key, val])
    dist_df = pd.DataFrame(dist_df, columns=['case', 'val'])
    
    loc_df = []
    for key, val in loc_dict.items():
        loc_df.append([key, val])
    loc_df = pd.DataFrame(loc_df, columns=['location', 'val'])
            
    return dist_df, loc_df

def load_global_age_distribution(
    data_path = './data/statistics/global_age.csv'
):
    age_df = pd.read_csv(data_path)
    age_dict = defaultdict(dict)
    
    for i, row in age_df.iterrows():
        country_gender = row['country_gender']
        country, gender = country_gender.split('_')
        
        temp_series = {
            '20-24': row['20-24'],
            '25-34': row['25-29'] + row['30-34'],
            '35-44': row['35-39'] + row['40-44'],
            '45-64': row['45-49'] + row['50-54'] + row['55-59'] + row['60-64'],
            '65-100': row['65-69'] + row['70-74'] + row['75-79'] + row['80-'],
        }

        temp_series = pd.Series(temp_series)
        age_dict[country][gender] = temp_series
            
    return age_dict

def load_global_education_distribution(
    data_path = './data/statistics/global_education.csv'
):
    education_df = pd.read_csv(data_path)
    
    education_dict = defaultdict(dict)
    
    for i, row in education_df.iterrows():
        country = row.values[0]
        female_education = pd.Series(
            row.values[1:5],
            index=['no_education', 'primary', 'secondary', 'tertiary']
        )
        female_education['primary or less'] = female_education['no_education'] + female_education['primary']
        female_education = female_education.drop(['no_education', 'primary'])
        female_education = female_education[['primary or less', 'secondary', 'tertiary']]
        
        male_education = pd.Series(
            row.values[5:9],
            index=['no_education', 'primary', 'secondary', 'tertiary']
        )
        male_education['primary or less'] = male_education['no_education'] + male_education['primary']
        male_education = male_education.drop(['no_education', 'primary'])
        male_education = male_education[['primary or less', 'secondary', 'tertiary']]
        
        education_dict[country]['Female'] = female_education
        education_dict[country]['Male'] = male_education
        
    return education_dict

def get_age(age_dict, origin, gender):
    if origin in [
        'Unknown', 'French Guiana', 'Palestinian', 'Guadeloupe', 'Macedonia',
        'Niue', 'Serbia',
    ]:
        age_dist = age_dict['World'][gender.lower()]
    else:
        age_dist = age_dict[origin][gender.lower()]
        
    age = random.choices(
        population=age_dist.index.tolist(),
        k=1,
        weights=[v for v in age_dist.values]
    )[0]
        
    return age

def get_education(education_dict, origin, gender):
    # If there is no information, we use average of the all countries.
    if origin in ['Unknown', 'Burkina Faso', 'Nigeria', 'Montenegro', 'Eritrea',
                   'Somalia', 'Chad', 'Georgia', 'Uzbekistan', 'Palestinian',
                   'Ethiopia', 'Guinea', 'Turkmenistan', 'Azerbaijan', 'Lebanon',
                   'Antigua', 'Angola', 'Equatorial Guinea', 'Cabo Verde', 'North Korea',
                   'St. Vincent and the Grenadines', 'Oman', 'Macedonia', 'Bhutan', 'Bosnia and Herzegovina',
                   'St. Lucia', 'Djibouti', 'Vanuatu', 'Guinea-Bissau', 'Niue',
                   'Guadeloupe', 'Antigua and Barbuda', 'Grenada', 'Bahamas', 'Madagascar',
                   'Seychelles', 'Suriname', 'Turks and Caicos Islands', 'French Guiana',
                   'Samoa', 'Cayman Islands', 'Comoros', 'Serbia']:
        if gender == 'Female':
            vals = [17.8, 23.7, 44.7, 13.8]
        elif gender == 'Male':
            vals = [12.0, 26.0, 48.5, 13.5]
            
        ref_vals = [vals[0] + vals[1], vals[2], vals[3]]
            
        educations = pd.Series(
            ref_vals,
            index=['primary or less', 'secondary', 'tertiary']
        )
    else:
        if origin in education_dict:
            educations = education_dict[origin][gender]
        elif origin == 'Congo':
            educations = education_dict["Republic of Congo"][gender]
        elif origin == 'South Sudan':
            educations = education_dict["Sudan"][gender]
        else:
            print(f'Country {origin} not in education_dict')

    education = random.choices(
        population=educations.index.tolist(),
        k=1,
        weights=[v for v in educations.values]
    )[0]
    
    return education

def solve_quadratic(
    mu_a,
    mu_c,
    mu_e,
    mu_g,
    tau_ceg,
    tau_aeg,
    tau_acg,
    tau_ace,
    rho_a=1.0,
    rho_c=1.0,
    rho_e=1.0,
    rho_g=1.0,
    rho_b=1.0,
    prob_range=(0.01, 0.99),
    exp=False
):
    dim_a = mu_a.shape[0]
    dim_c = mu_c.shape[0]
    dim_e = mu_e.shape[0]
    dim_g = mu_g.shape[0]

    dim_total = dim_a * dim_c * dim_e * dim_g

    R = np.eye(dim_total)
    
    s = np.ones((dim_a, dim_c, dim_e, dim_g))
    for a in range(dim_a):
        for c in range(dim_c):
            for e in range(dim_e):
                for g in range(dim_g):
                    s[a, c, e, g] = dim_a * mu_a[a] + dim_c * mu_c[c] + dim_e * mu_e[e] + dim_g * mu_g[g]
    s = s / (dim_a + dim_c + dim_e + dim_g)
    s = s.flatten()
    
    A_a = []
    b_a = np.zeros(dim_a)
    for a in range(dim_a):
        A_sub = np.zeros((dim_a, dim_c, dim_e, dim_g))
        for c in range(dim_c):
            for e in range(dim_e):
                for g in range(dim_g):
                    A_sub[a, c, e, g] = tau_ceg[c, e, g]
        A_sub = A_sub.flatten()
        A_sub = np.expand_dims(A_sub, axis=0)
        A_a.append(A_sub)
        b_a[a] = mu_a[a]
    A_a = np.concatenate(A_a, axis=0)
    
    G_au = A_a
    G_al = A_a * -1
    
    if exp:
        h_au = b_a ** (1 / rho_a)
        h_al = b_a ** (1 * rho_a) * -1
    else:
        h_au = b_a * (1 + rho_a)
        h_al = b_a * (1 - rho_a) * -1
    
    G_a = np.concatenate([G_au, G_al], axis=0)
    h_a = np.concatenate([h_au, h_al], axis=0)
    
    
    A_c = []
    b_c = np.zeros(dim_c)
    for c in range(dim_c):
        A_sub = np.zeros((dim_a, dim_c, dim_e, dim_g))
        for a in range(dim_a):
            for e in range(dim_e):
                for g in range(dim_g):
                    A_sub[a, c, e, g] = tau_aeg[a, e, g]
        A_sub = A_sub.flatten()
        A_sub = np.expand_dims(A_sub, axis=0)
        A_c.append(A_sub)
        b_c[c] = mu_c[c]
    A_c = np.concatenate(A_c, axis=0)
    
    G_cu = A_c
    G_cl = A_c * -1
    
    if exp:
        h_cu = b_c ** (1 / rho_c)
        h_cl = b_c ** (1 * rho_c) * -1
    else:
        h_cu = b_c * (1 + rho_c)
        h_cl = b_c * (1 - rho_c) * -1
    
    G_c = np.concatenate([G_cu, G_cl], axis=0)
    h_c = np.concatenate([h_cu, h_cl], axis=0)
    
    
    A_e = []
    b_e = np.zeros(dim_e)
    for e in range(dim_e):
        A_sub = np.zeros((dim_a, dim_c, dim_e, dim_g))
        for a in range(dim_a):
            for c in range(dim_c):
                for g in range(dim_g):
                    A_sub[a, c, e, g] = tau_acg[a, c, g]
        A_sub = A_sub.flatten()
        A_sub = np.expand_dims(A_sub, axis=0)
        A_e.append(A_sub)
        b_e[e] = mu_e[e]
    A_e = np.concatenate(A_e, axis=0)
    
    G_eu = A_e
    G_el = A_e * -1
    
    if exp:
        h_eu = b_e ** (1 / rho_e)
        h_el = b_e ** (1 * rho_e) * -1
    else:
        h_eu = b_e * (1 + rho_e)
        h_el = b_e * (1 - rho_e) * -1
    
    G_e = np.concatenate([G_eu, G_el], axis=0)
    h_e = np.concatenate([h_eu, h_el], axis=0)
    
    
    A_g = []
    b_g = np.zeros(dim_g)
    for g in range(dim_g):
        A_sub = np.zeros((dim_a, dim_c, dim_e, dim_g))
        for a in range(dim_a):
            for c in range(dim_c):
                for e in range(dim_e):
                    A_sub[a, c, e, g] = tau_ace[a, c, e]
        A_sub = A_sub.flatten()
        A_sub = np.expand_dims(A_sub, axis=0)
        A_g.append(A_sub)
        b_g[g] = mu_g[g]
    A_g = np.concatenate(A_g, axis=0)
    
    G_gu = A_g
    G_gl = A_g * -1
    
    if exp:
        h_gu = b_g ** (1 / rho_g)
        h_gl = b_g ** (1 * rho_g) * -1
    else:
        h_gu = b_g * (1 + rho_g)
        h_gl = b_g * (1 - rho_g) * -1
    
    G_g = np.concatenate([G_gu, G_gl], axis=0)
    h_g = np.concatenate([h_gu, h_gl], axis=0)
    
    G_ub = []
    h_ub = []
    h_lb = []

    for a in range(dim_a):
        for c in range(dim_c):
            for e in range(dim_e):
                for g in range(dim_g):
                    G_sub = np.zeros((dim_a, dim_c, dim_e, dim_g))
                    G_sub[a, c, e, g] = 1
                    G_sub = np.expand_dims(G_sub.flatten(), axis=0)
                    G_ub.append(G_sub)
                    
                    h_ub_sub = np.array([mu_a[a], mu_c[c], mu_e[e], mu_g[g]])
                    if exp:
                        h_ub_sub = h_ub_sub ** (1 / rho_b)
                    else:
                        h_ub_sub = h_ub_sub * (1 + rho_b)
                    h_ub_sub = h_ub_sub.min()
                    h_ub_sub = min(prob_range[1], h_ub_sub)
                    h_ub.append(h_ub_sub)
                    
                    h_lb_sub = np.array([mu_a[a], mu_c[c], mu_e[e], mu_g[g]])
                    if exp:
                        h_lb_sub = h_lb_sub ** (1 * rho_b)
                    else:
                        h_lb_sub = h_lb_sub * (1 - rho_b)
                    h_lb_sub = h_lb_sub.max()
                    h_lb_sub = max(prob_range[0], h_lb_sub)
                    h_lb.append(h_lb_sub)
                    
    G_ub = np.concatenate(G_ub, axis=0)
    h_ub = np.array(h_ub)
    
    G_lb = G_ub * -1
    h_lb = np.array(h_lb) * -1
    
    G_b = np.concatenate([G_ub, G_lb], axis=0)
    h_b = np.concatenate([h_ub, h_lb], axis=0)

    A = np.concatenate([A_a, A_c, A_e, A_g], axis=0)
    b = np.concatenate([b_a, b_c, b_e, b_g], axis=0)
    
    G = np.concatenate([G_a, G_c, G_e, G_g, G_b], axis=0)
    h = np.concatenate([h_a, h_c, h_e, h_g, h_b], axis=0)
    
    try:
        sol = solve_ls(
            R=R,
            s=s,
            G=G,
            h=h,
            verbose=False,
            solver='osqp',
        )
    except Exception as e:
        sol = None
        
    if type(sol) == np.ndarray:
        sol = np.clip(sol, prob_range[0], prob_range[1])
        sol = sol.reshape((dim_a, dim_c, dim_e, dim_g))

    return sol

def synthesize_refugee(
    refugee_batch_size=100,
    refugee_batch_num=5000,
    location_num=10,
    seed=0
):
    reset_seed(seed)
    
    location = LOCATIONS[:location_num]
    dist_df, loc_df = load_refugee_distribution()
    
    loc_df = loc_df.loc[loc_df['location'].isin(location)].reset_index(drop=True)
    
    age_dict = load_global_age_distribution()
    education_dict = load_global_education_distribution()
    
    refugee_df = []
    for b in tqdm(range(refugee_batch_num), desc='Generating Refugee Details'):
        origin_gender_list = random.choices(
            population=dist_df['case'].tolist(),
            k=refugee_batch_size,
            weights=dist_df['val'].tolist()
        )
        
        for i, origin_gender in enumerate(origin_gender_list):
            refugee_id = f'RF.{b:05}.{i:04}'
            
            origin, gender = origin_gender.split('_')
            age = get_age(age_dict, origin, gender)
            education = get_education(education_dict, origin, gender)

            refugee_df.append([refugee_id, origin, age, education, gender])

    refugee_df = pd.DataFrame(refugee_df, columns = ['refugee_id'] + FIELDS)
    
    return refugee_df, loc_df

def synthesize_capacity(
    loc_df,
    refugee_batch_size=100,
    refugee_batch_num=5000,
    location_num=10,
    seed=0
):
    def get_prob_capacity(loc_df, refugee_batch_size):
        vals = loc_df['val'].to_numpy()
        
        cur_capacity = random.choices(
            population=loc_df['location'],
            k=refugee_batch_size,
            weights=vals
        )
        
        return cur_capacity
    
    reset_seed(seed)
    
    locations = LOCATIONS[:location_num]
        
    capacity_df = []
    for b in range(refugee_batch_num):
        cur_capacity = get_prob_capacity(loc_df, refugee_batch_size)
        loc_counter = Counter(cur_capacity)
        
        capacity = [loc_counter[loc] if loc in loc_counter else 0 for loc in locations]
        capacity_df.append(capacity)
    capacity_df = pd.DataFrame(capacity_df, columns=locations)
    
    return capacity_df


def synthesize_location_probs(
    refugee_df,
    location_num,
    var=0.001,
    data_path='./data/statistics',
    seed=0,
):
    reset_seed(seed)
    dp = Path(data_path)
    
    n_aeg_l = np.load(dp / 'n_aeg_l.npy') # (A, E, G, L)
    dim_a, dim_e, dim_g, dim_l = n_aeg_l.shape
    tau_w_lf = np.load(dp / 'tau_w_lf.npy') # (L)
    
    tau_a_lw = np.load(dp / 'tau_a_lw.npy') # (A, L)
    n_a_l = n_aeg_l.sum(axis=(1, 2)) 
    tau_a_l = n_a_l / n_a_l.sum(axis=0) 
    mu_a_l = tau_a_lw * np.expand_dims(tau_w_lf, axis=0) / tau_a_l # (A, L)
    
    tau_c_lwf = np.load(dp / 'tau_c_lwf.npy') # (C, L)
    dim_c, _ = tau_c_lwf.shape
    tau_c_lf = np.load(dp / 'tau_c_lf.npy', allow_pickle=True) # (C, L)
    mu_c_l = tau_c_lwf * np.expand_dims(tau_w_lf, axis=0) / tau_c_lf # (C, L)
    
    tau_e_lwf = np.load(dp / 'tau_e_lwf.npy') # (E, L)
    tau_e_lf = np.load(dp / 'tau_e_lf.npy') # (E, L)
    mu_e_l = tau_e_lwf * np.expand_dims(tau_w_lf, axis=0) / tau_e_lf # (E, L)
    
    tau_g_lw = np.load(dp / 'tau_g_lw.npy') # (G, L)
    tau_g_lf = np.load(dp / 'tau_g_lf.npy') # (G, L)
    mu_g_l = tau_g_lw * np.expand_dims(tau_w_lf, axis=0) / tau_g_lf # (G, L)
    
    n_eg_l = n_aeg_l.sum(axis=0) # (E, G, L)
    tau_eg_l = n_eg_l / n_eg_l.sum(axis=(0, 1))
    tau_ceg_lf = np.expand_dims(tau_c_lf, axis=(1, 2)) * np.expand_dims(tau_eg_l, axis=0) # (C, E, G, L)
    
    tau_aeg_lf = n_aeg_l / n_aeg_l.sum(axis=(0, 1, 2)) # (A, E, G, L)
    
    n_ag_l = n_aeg_l.sum(axis=1) # (A, G, L)
    tau_ag_l = n_ag_l / n_ag_l.sum(axis=(0, 1))
    tau_acg_lf = np.expand_dims(tau_c_lf, axis=(0, 2)) * np.expand_dims(tau_ag_l, axis=1) # (A, C, G, L)
        
    n_ae_l = n_aeg_l.sum(axis=2) # (A, E, L)
    tau_ae_l = n_ae_l / n_ae_l.sum(axis=(0, 1))
    tau_ace_lf = np.expand_dims(tau_c_lf, axis=(0, 2)) * np.expand_dims(tau_ae_l, axis=1) # (A, C, E, L)
    
    mu_aceg_l = []
        
    for location in range(location_num):
        m_target = solve_quadratic(
            mu_a=mu_a_l[:, location],
            mu_c=mu_c_l[:, location],
            mu_e=mu_e_l[:, location],
            mu_g=mu_g_l[:, location],
            tau_ceg=tau_ceg_lf[:, :, :, location],
            tau_aeg=tau_aeg_lf[:, :, :, location],
            tau_acg=tau_acg_lf[:, :, :, location],
            tau_ace=tau_ace_lf[:, :, :, location],
            rho_a=0.5,
            rho_c=0.0,
            rho_e=0.1,
            rho_g=0.0,
            rho_b=0.6,
        )

        mu_aceg_l.append(m_target)
        
    mu_aceg_l = np.stack(mu_aceg_l, axis=-1)
    
    concentration_alpha, concentration_beta = calculate_beta_distribution_parameter(mu_aceg_l, var)

    location_probs = []
    
    empirical_mu_aceg_l = []
    for i, row in tqdm(refugee_df.iterrows(), total=len(refugee_df), desc='Calculating Scores'):
        origin = row['origin']
        age = row['age']
        education = row['education']
        gender = row['gender']
        
        # Get base probability
        min_age, max_age = age.split('-')
        avg_age = (float(min_age) + float(max_age)) / 2
        
        age_group = None
        age_index = -1
        for s in AGE_GROUPS:
            lb, ub = s.split('-')
            if avg_age >= int(lb) and avg_age <= int(ub):
                age_group = s
                age_index = list(AGE_GROUPS).index(s)
        assert(age_group != None)
        
        continent = get_continent(origin)
        continent_index = list(CONTINENT_GROUPS).index(continent)
        
        education_index = list(EDUCATION_GROUPS).index(education)
        gender_index = list(GENDER_GROUPS).index(gender)
                
        concentration_alpha_l = concentration_alpha[age_index, continent_index, education_index, gender_index]
        concentration_beta_l = concentration_beta[age_index, continent_index, education_index, gender_index]
        
        mu_i = mu_aceg_l[age_index, continent_index, education_index, gender_index]
        
        emp_prob = np.random.beta(concentration_alpha_l, concentration_beta_l)
        location_probs.append(emp_prob)
        empirical_mu_aceg_l.append(mu_i)
        
    location_probs = np.stack(location_probs, axis=0)
    empirical_mu_aceg_l = np.stack(empirical_mu_aceg_l, axis=0)
    
    location_probs = np.clip(location_probs, a_min=1e-6, a_max=None)
    
    return location_probs, mu_aceg_l, empirical_mu_aceg_l

def synthesize_employment(
    location_probs,
    refugee_batch_size=100,
    refugee_batch_num=5000,
    location_num=10,
    seed=0
):
    reset_seed(seed)
    
    locations = LOCATIONS[:location_num]
    probs = location_probs.flatten()
    
    employments = []
    for prob in tqdm(probs, desc='Generating Employments'):
        emp = random.choices(
            population=[0, 1],
            k=1,
            weights=[1 - prob, prob]
        )[0]
        employments.append(emp)
        
    employments = np.array(employments)
    employments = employments.reshape(location_probs.shape)
    
    return employments

def shuffle_assignment(
    assignment,
    noise_ratio,
    seed=0
):
    reset_seed(seed)
    
    assignment = copy.deepcopy(assignment)
    n_batch, batch_size = assignment.shape
    
    if noise_ratio == 0.0:
        return assignment
    
    n = int(batch_size * noise_ratio)
    
    new_assignment = []
    for i, batch_assignment in enumerate(assignment):
        victim_idx = np.array(random.sample(range(batch_size), n))
        victim_loc = batch_assignment[victim_idx]
        random.shuffle(victim_loc)

        for j, vi in enumerate(victim_idx):
            batch_assignment[vi] = victim_loc[j]
            
        new_assignment.append(batch_assignment)

    new_assignment = np.stack(new_assignment, axis=0)

    return new_assignment