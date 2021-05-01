import pandas as pd
import numpy as np
import time
from pathos.pools import ProcessPool


class AlsSolver:

    def __init__(self, R:np.array, U:np.array, V:np.array, tol:float, lambda_u:float, lambda_i:float, njobs:int):
                
        self.R = R
        self.U = U
        self.V = V
        self.k = self.U.shape[1]
        self.lambda_u = lambda_u
        self.lambda_i = lambda_i
        self.tol = tol
        self.njobs = njobs
        self.run_parallel = True if njobs>1 else False
        self.U_split=None
        self.V_split=None

        # Creates splits of U and V if run_parallel is True
        if self.run_parallel:
            self.__create_parallel_splits()


    def __format_matrices(self):
        pass

    
    def __solve_users(self,U:np.array, V:np.array, R:np.array):
        # import pdb;pdb.set_trace()
        users = R['user'].sort_values(ascending=True).unique()
        U_new = []
        # Let's iterate over our users
        for i in users:

            
        
            # Let's get our user id
            user_i_id = i

            # Interaction matrix for user user_i_id
            Ri = R[R['user']==user_i_id].sort_values(by='item',ascending=True)

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

            # Save embedding
            U_i = np.reshape(
                # Compute pseudo-inverse of Viobs times Riobs
                np.matmul(
                    A,
                    Riobs
                ),
                (1,self.k)
            )
            U_new.append(U_i)
            
        return np.array(U_new).reshape((users.shape[0],self.k))


    def __solve_items(self,U:np.array, V:np.array, R:np.array):
        
        items = R['item'].sort_values(ascending=True).unique()
        # items.sort()
        V_new = []
        for j in items:
        
            # Let's get our item id
            item_j_id = j

            # Interaction matrix for item item_j_id
            Rj = R[R['item']==item_j_id].sort_values(by='user',ascending=True)

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

            # Save embedding
            V_j = np.reshape(
                # Compute pseudo-inverse of Ujobs times Rjobs
                np.matmul(
                    A,
                    Rjobs
                ),
                (1,self.k)
            )
            V_new.append(V_j)

        return np.array(V_new).reshape((items.shape[0],self.k))


    def __calculate_error(self):
        
        self.R['interaction_predict'] = [np.dot(
                                            self.U[tuple.user],
                                            self.V[tuple.item]
                                        ) for tuple in self.R.itertuples()]
        self.R['err'] = np.square(self.R['interaction']-self.R['interaction_predict'])

        return self.R['err'].mean()


    def __solve_iter(self):
        
        print('Solving users..')
        self.U = self.__solve_users(
            U=self.U,
            V=self.V,
            R=self.R
        )
        print('Done')
        print('Solving items..')
        self.V = self.__solve_items(
            U=self.U,
            V=self.V,
            R=self.R
        )
        print('Done')


    def __create_parallel_splits(self):

        # Creates user splits
        U_splits = np.array_split(
            ary=self.U,
            indices_or_sections=self.njobs,
            axis=0
        )
        # Gets user index boundaries in each split and obtains equivalent R matrices
        U_splits_indexes = np.cumsum([0] + [x.shape[0] for x in U_splits])
        R_user_splits = [self.R[(self.R['user']>=U_splits_indexes[split_n-1])
                                &(self.R['user']<U_splits_indexes[split_n])] for split_n in range(1,U_splits_indexes.shape[0])]


        # Creates (U,R) tuples for each split and saves ParallelSplit objects
        U_R_splits = zip(U_splits, R_user_splits)                       
        self.U_split = [
            ParallelSplit(
                R=tup[1],
                embs=tup[0],
                order=i
            ) for i, tup in enumerate(U_R_splits)
        ]

        # Creates item splits
        V_splits = np.array_split(
            ary=self.V,
            indices_or_sections=self.njobs,
            axis=0
        )
        # Gets item index boundaries in each split and obtains equivalent R matrices
        V_splits_indexes = np.cumsum([0] + [x.shape[0] for x in V_splits])
        R_item_splits = [self.R[(self.R['item']>=V_splits_indexes[split_n-1])
                                &(self.R['item']<V_splits_indexes[split_n])] for split_n in range(1,V_splits_indexes.shape[0])]
        
        
        # Creates (U,R) tuples for each split and saves ParallelSplit objects
        V_R_splits = zip(V_splits, R_item_splits)                       
        self.V_split = [
            ParallelSplit(
                R=tup[1],
                embs=tup[0],
                order=i
            ) for i,tup in enumerate(V_R_splits)
        ]
        


    def __solve_iter_parallel(self):

        def solve_users_parallel(split:ParallelSplit):

            U_new = self.__solve_users(
                U=split.embs,
                V=self.V,
                R=split.R
            )
            return split.set_embs(np.array(U_new))

        def solve_items_parallel(split:ParallelSplit):
            V_new = self.__solve_items(
                U=self.U,
                V=split.embs,
                R=split.R
            )
            return split.set_embs(np.array(V_new))

        

        pool = ProcessPool(nodes=self.njobs)

        # Let's process and update user embeddings
        print('Solving users..')
        U_splits = pool.map(solve_users_parallel, self.U_split)
        U_new = np.empty(shape=(0,self.k))
        for u_split in U_splits:
            u_split.set_embs(u_split.embs.reshape(u_split.embs.shape[0],self.k))
            U_new = np.concatenate(
                (U_new, u_split.embs)
            )
        self.U = U_new
        print('Done')

        # Let's process and update item embeddings
        print('Solving items..')
        V_splits = pool.map(solve_items_parallel, self.V_split)
        V_new = np.empty(shape=(0,self.k))
        for v_split in V_splits:
            v_split.set_embs(v_split.embs.reshape(v_split.embs.shape[0],self.k))
            V_new = np.concatenate(
                (V_new, v_split.embs)
            )
        self.V = V_new
        print('Done')

        # Updates splits
        self.U_split = U_splits
        self.V_split = V_splits


    def solve(self):
        
        mse = []
        new_mse=self.__calculate_error()
        previous_mse = 2*new_mse
        n=1
        start_time = time.time()
        new_time = start_time
        while abs(new_mse-previous_mse)/previous_mse > self.tol:
            print('-'*100)
            print('== ITERATION {} =='.format(n))
            
            if not self.run_parallel:
                self.__solve_iter()
            elif self.run_parallel:
                self.__solve_iter_parallel()
            
            previous_mse = new_mse
            print('Calculating prediction error..')
            new_mse = self.__calculate_error()
            print('Done')
            print('MSE: {}'.format(round(new_mse,3)))
            mse.append(new_mse)
            n+=1
            previous_time = new_time
            new_time = time.time()
            print('Total elapsed time: {}s'.format(round(new_time-start_time,1)))
            print('Time in this iteration: {}s'.format(round(new_time-previous_time,1)))

        print('-'*100)

        return self.U, self.V, np.array(mse)
 


class ParallelSplit:

    def __init__(self, R:pd.DataFrame, embs:np.array, order:int):
        self.R=R
        self.embs=embs
        self.order=order

    def set_embs(self,embs):
        self.embs=embs
        return self