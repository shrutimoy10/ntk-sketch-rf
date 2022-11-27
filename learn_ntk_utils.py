import numpy as np



def computeClassificationError(Z, w, mean_y, y):
    '''
    Computes the classification errors for ridge regression models
    that are trained using zero-mean label vectors
    Input : 
    Z : random feature matrix
    w : optimized classification parameter vector
    mean_y : mean(y_train)
    y : true labels for datapoints in Z
    '''
    pred = np.sign(np.dot(np.transpose(Z), w) + mean_y)
    error = np.mean( pred != y)
    fn = np.mean(pred!=y and y == 1)/error
    fp = np.mean(pred!=y and y == -1)/error

    return error, fn, fp


def createRandomFourierFeatures(D, W, b, X):
    '''
    Creates Gaussian random features
    Input:
    D : number of features 
    W, b : learnable parameters for those features
    X : dataset
    '''
    return np.sqrt(2/D) * np.cos(np.dot(np.transpose(W),X)+b)



def project_onto_simplex(v, B):
    '''
    Projects a vector v onto a scaled probability simplex
    defined by the set
        {x >= 0 | sum(x) == B}
    The projection is computed using Euclidean distance
    '''
    # print("v shape : ", v.shape)

    u = np.sort(v)
    u = u[::-1]  # sorting in descending order
    sv = np.cumsum(u)
    lv = np.arange(1, len(v)+1, 1)
    rho = np.argwhere(u > (sv - B)/lv)
    rho = rho[-1]
    theta = (sv[rho] - B) / (rho+1)

    return np.maximum(v - theta, 0)


def linear_chi_square(v, u, rho , acc = 1e-8):
    '''
    Returns a probability distribution x that solves the 
    quadratically constrained linear problem
            max x.T * v
            s.t  ||x - u||^2 <= rho,
                sum(x) = 1,
                x >= 0
    Uses a binary search strategy along with projections onto the simplex to
    solve the problem in O(n log (n / acc)) time to solve to accuracy acc (in
    duality gap) -- Algorithm 1 in "Learning Kernels with Random Features" paper.
    '''
    duality_gap = np.inf

    max_lambda = np.inf
    min_lambda = 0

    #Projecting onto the probability simplex
    x = project_onto_simplex(u,1) 

    if(np.linalg.norm(x - u, ord = 2)**2 > rho):
       print('Problem is not feasible')
       return
    
    start_lambda = 1

    while(np.isinf(max_lambda)):
        # print("while u shape : {}".format(u.shape))
        # print("while v/start_lambda shape : {}".format((v/start_lambda).shape))
        # print("shape : {}".format((u - v/start_lambda).shape))
        x = project_onto_simplex(u - v/start_lambda, 1)
        lam_grad = 0.5 * np.linalg.norm((x - u), ord = 2)**2 - rho / 2
        if(lam_grad < 0):
            max_lambda = start_lambda
        else: 
            start_lambda *= 2
    
    while(max_lambda - min_lambda > acc * start_lambda):
        lam = (min_lambda + max_lambda) / 2
        x = project_onto_simplex(u - v/lam, 1)
        lam_grad = 0.5 * np.linalg.norm((x - u), ord = 2)**2 - rho / 2
        if(lam_grad < 0):
            max_lambda = lam
        else:
            min_lambda = lam

    return x

