import numpy as np
from learn_ntk_utils import linear_chi_square

def project_onto_simplex(v, B):

    u = np.sort(v)
    u = u[::-1]  # sorting in descending order
    print("\n u = ", u)
    sv = np.cumsum(u)
    print("\n sv = ", sv)
    lv = np.arange(1, len(v)+1, 1)
    print("\n lv = ", lv)
    print("\n (sv - B)/lv = ", (sv - B)/(lv))
    rho = np.argwhere(u > ((sv - B)/(lv)))
    print("rho arr : ", rho)
    rho = rho[-1]
    print("\n rho : ", rho)
    theta = (sv[rho] - B) / (rho+1)
    print("\n Theta : ",theta)

    print("v - theta", v - theta)
    return np.maximum(v - theta, 0)


# def linear_chi_square(v, u, rho , acc = 1e-8):
#     '''
#     Returns a probability distribution x that solves the 
#     quadratically constrained linear problem
#             max x.T * v
#             s.t  ||x - u||^2 <= rho,
#                 sum(x) = 1,
#                 x >= 0
#     Uses a binary search strategy along with projections onto the simplex to
#     solve the problem in O(n log (n / acc)) time to solve to accuracy acc (in
#     duality gap) -- Algorithm 1 in "Learning Kernels with Random Features" paper.
#     '''
#     duality_gap = np.inf

#     max_lambda = np.inf
#     min_lambda = 0

#     #Projecting onto the probability simplex
#     x = project_onto_simplex(u,1) 

#     if(np.linalg.norm(x - u, ord = 2)**2 > rho):
#        print('Problem is not feasible')
#        return
    
#     start_lambda = 1

#     while(np.isinf(max_lambda)):
#         x = project_onto_simplex(u - v/start_lambda, 1)
#         lam_grad = 0.5 * np.linalg.norm((x - u), ord = 2)**2 - rho / 2
#         if(lam_grad < 0):
#             max_lambda = start_lambda
#         else: 
#             start_lambda *= 2
    
#     while(max_lambda - min_lambda > acc * start_lambda):
#         lam = (min_lambda + max_lambda) / 2
#         x = project_onto_simplex(u - v/lam, 1)
#         lam_grad = 0.5 * np.linalg.norm((x - u), ord = 2)**2 - rho / 2
#         if(lam_grad < 0):
#             max_lambda = lam
#         else:
#             min_lambda = lam

#     return x


Nw = 10
rho = Nw * 0.005

v = np.random.rand(Nw)
print("\n v : ", v)

u = np.ones(Nw) * (1/Nw)

# B = 1
# x = project_onto_simplex(v, B)

x = linear_chi_square(v, u, rho , 1e-8)


print('x : ',x)
print(np.sum(x))