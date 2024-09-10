import numpy as np

'''
@brief This function computes the Gram Schmidt Orthogonalization of a given basis. It also gives the projection matrix as well as the squared lengths of the 
new gram schmidt basis vectors. 
'''
def GramSchmidt(basis):
    basis = np.array(basis)
    M = basis.shape[0]
    projection_mat = np.zeros((M,M))

    basis = basis.astype(np.float64)
    gram_basis = np.ones_like(basis).astype(np.float64)
    gram_basis[0] = basis[0]
    squared_lengths = np.zeros((M,1))
    squared_lengths[0] = np.dot(gram_basis[0],gram_basis[0])
    for i in range(1,basis.shape[0]):
        gram_basis[i] = basis[i]
        projection = 0
        proj_vec = np.zeros_like(gram_basis[0])
        for j in range(i):
            projection = (np.dot(gram_basis[i],gram_basis[j]))/(np.dot(gram_basis[j],gram_basis[j]))
            projection_mat[i,j] = projection
            
            #print(projection)
            proj2_vec = (projection*gram_basis[j])
            
            #print(proj2_vec)
            proj_vec = proj_vec +proj2_vec
      
        gram_basis[i] = gram_basis[i]-proj_vec
        squared_lengths[i] = np.dot(gram_basis[i],gram_basis[i])
    np.fill_diagonal(projection_mat,1)
    
    return gram_basis,projection_mat,squared_lengths

'''
@brief This function performs the reduction operation of the LLL algorithm that is it checks if the projection mat is within range and 
accordingly alters the vectors in the given basis. 
'''
def reduce(k:int,l:int,projection_mat,new_basis):
    new_basis_copy = new_basis.copy()
    projection_mat_copy = projection_mat.copy()
  
    if np.abs(projection_mat[k,l]) > 0.5:
        new_basis[k] = new_basis_copy[k]-np.ceil(projection_mat_copy[k,l]-0.5)*new_basis_copy[l]
        
        for j in range(0,l):
            
            projection_mat[k,j] = projection_mat_copy[k,j]-np.ceil(projection_mat_copy[k,l]-0.5)*projection_mat_copy[l,j]
        projection_mat[k,l] = projection_mat_copy[k,l] - np.ceil(projection_mat_copy[k,l]-0.5)
    return projection_mat,new_basis

'''
@brief This function performs the exchange operation of the LLL algorithm that is it exchanges two rows in the basis matrix based on the 
exchange condition. 
'''
def exchange(k:int,new_basis,projection_mat,squared_lengths):
    new_basis_copy =  new_basis.copy()
    projection_mat_copy = projection_mat.copy()
    squared_lengths_copy = squared_lengths.copy()


    new_basis[k-1] = new_basis_copy[k]
    new_basis[k] = new_basis_copy[k-1]
    v = projection_mat_copy[k,k-1]
    delta = squared_lengths[k] + (v**2)*squared_lengths[k-1]
    projection_mat[k,k-1] = (v*squared_lengths[k-1])/(delta) 
    squared_lengths[k] = (squared_lengths[k]*squared_lengths[k-1])/(delta)
    squared_lengths[k-1] = delta
    for j in range(0,k-1): #not sure if it should be k-1
        t = projection_mat[k-1,j]
        projection_mat[k-1,j] = projection_mat[k,j]
        projection_mat[k,j] = t
    for i in range(k+1,projection_mat.shape[1]):
        epsilon = projection_mat[i,k]
        projection_mat[i,k] = projection_mat[i,k-1] - v*projection_mat[i,k]
        projection_mat[i,k-1] = projection_mat[k,k-1]*projection_mat[i,k] + epsilon 
    return squared_lengths,projection_mat,new_basis

'''
@brief This function computes the LLL reduced basis of a given basis and alpha value (1/4<=alpha<=1)
'''
def calc_LLL_basis(basis,alpha):
    basis = np.array(basis)
    LLL_basis = basis.astype(np.float64)
    gram_basis,projection_mat,squared_lengths = GramSchmidt(LLL_basis)
    k = 1
    inumb = 1
    LLL_iter = np.zeros((0,basis.shape[0],basis.shape[1]))
    LLL_iter = np.append(LLL_iter,LLL_basis[np.newaxis,:,:],axis = 0)
    while (k<=LLL_basis.shape[0]-1):
        projection_mat,LLL_basis = reduce(k,k-1,projection_mat,LLL_basis)
        if squared_lengths[k] >= (alpha-(projection_mat[k,k-1]**2))*squared_lengths[k-1]:
            for l in reversed(range(k-1)):
                projection_mat,LLL_basis = reduce(k,l,projection_mat,LLL_basis)
            k = k+1
        else:
            squared_lengths,projection_mat,LLL_basis = exchange(k,LLL_basis,projection_mat,squared_lengths)
            if k>1:
                k = k-1
        LLL_iter = np.append(LLL_iter,LLL_basis[np.newaxis,:,:],axis = 0)
        # print('LLL_basis:',LLL_basis)
        # print('projection_mat:',projection_mat)
        # print('squared_lengths:',squared_lengths)
        # print(inumb)
        
        inumb +=1
    return LLL_basis,LLL_iter,inumb-1



