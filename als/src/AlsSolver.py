import pandas as pd
import numpy as np
from tqdm import tqdm
import time


class AlsSolver:

    def __init__(self, R:np.array, U:np.array, V:np.array, tol:float, lambda_u:float, lambda_i:float, U_order:int=None):
        # if (U.shape[0],V.shape[1]) != R.shape:
        #     raise AssertionError('Shape {r_shape} of user-item interaction matrix R is not compatible with\
        #                           user embedding matrix of shape {u_shape} and item embedding matrix of shape {v_shape}'\
        #                               .format(r_shape=R.shape, u_shape = U.shape, v_shape = V.shape))
        
        self.R = R
        self.U = U
        self.V = V
        self.U_order = U_order
        self.k = self.U.shape[1]
        self.lambda_u = lambda_u
        self.lambda_i = lambda_i
        self.tol=tol

    def __format_matrices(self):
        pass

    
    def __solve_users(self,U:np.array, V:np.array, R:np.array):
        
        users = R['user'].unique()
        for i in tqdm(range(users.shape[0])):
        
            # Let's get our user id
            user_i_id = users[i]

            # Interaction matrix for user user_i_id
            Ri = R[R['user']==user_i_id]

            # Let's get our observed item embeddings and interaction values
            observed_item_idx = Ri['item'].values
            niobs = len(observed_item_idx)
            Viobs = np.take(
                a=V,
                indices=observed_item_idx,
                axis=0
            )
            Riobs = Ri['interaction'].values

            # Reshape matrices
            Viobs = np.reshape(Viobs,(niobs,self.k))
            Riobs = np.reshape(Riobs,(niobs,1))

            # Calculates solution matrix
            A = np.linalg.pinv(
                np.matmul(
                    np.transpose(Viobs),
                    Viobs
                ) + self.lambda_u*np.identity(self.k)
            )
            A = np.matmul(
                A,
                np.transpose(Viobs)
            )

            # Update embedding
            self.U[i] = np.reshape(
                # Compute pseudo-inverse of Viobs times Riobs
                np.matmul(
                    A,
                    Riobs
                ),
                (1,self.k)
            )

    def __solve_items(self,U:np.array, V:np.array, R:np.array):
        
        items = R['item'].unique()
        items.sort()
        for j in tqdm(range(items.shape[0])):
        
            # Let's get our item id
            item_j_id = items[j]

            # Interaction matrix for item item_j_id
            Rj = R[R['item']==item_j_id]

            # Let's get our observed user embeddings and interaction values
            observed_user_idx = Rj['user'].values
            nuobs = len(observed_user_idx)
            Ujobs = np.take(
                a=U,
                indices=observed_user_idx,
                axis=0
            )
            Rjobs = Rj['interaction'].values

            # Reshape matrices
            Ujobs = np.reshape(Ujobs,(nuobs,self.k))
            Rjobs = np.reshape(Rjobs,(nuobs,1))

            # Calculates solution matrix
            A = np.linalg.pinv(
                np.matmul(
                    np.transpose(Ujobs),
                    Ujobs
                ) + self.lambda_i*np.identity(self.k)
            )
            A = np.matmul(
                A,
                np.transpose(Ujobs)
            )

            # Update embedding
            self.V[j] = np.reshape(
                # Compute pseudo-inverse of Ujobs times Rjobs
                np.matmul(
                    # np.linalg.pinv(Ujobs),
                    A,
                    Rjobs
                ),
                (1,self.k)
            )

    def __calculate_error(self):
        
        self.R['interaction_predict'] = [np.dot(
                                            self.U[tuple.user],
                                            self.V[tuple.item]
                                        ) for tuple in self.R.itertuples()]
        self.R['err'] = np.square(self.R['interaction']-self.R['interaction_predict'])

        return self.R['err'].mean()


    def solve(self):
        
        mse = []
        new_mse=self.__calculate_error()
        previous_mse = 2*new_mse
        n=1
        start_time = time.time()
        new_time = start_time
        while abs(new_mse-previous_mse) > self.tol:
            print('-'*100)
            print('== ITERAÇÃO {} =='.format(n))
            print('Resolvendo usuarios..')
            self.__solve_users(
                U=self.U,
                V=self.V,
                R=self.R
            )
            print('Feito')
            print('Resolvendo itens..')
            self.__solve_items(
                U=self.U,
                V=self.V,
                R=self.R
            )
            print('Feito')
            previous_mse = new_mse
            print('Calculando erro de predição..')
            new_mse = self.__calculate_error()
            print('Feito')
            print('MSE: {}'.format(round(new_mse,3)))
            mse.append(new_mse)
            n+=1
            previous_time = new_time
            new_time = time.time()
            print('Total elapsed time: {}s'.format(round(new_time-start_time,1)))
            print('Time in this iteration: {}s'.format(round(new_time-previous_time,1)))

        print('-'*100)

        return self.U, self.V, np.array(mse)
 
