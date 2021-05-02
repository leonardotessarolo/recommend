import pandas as pd
import numpy as np
import pickle

from .AlsSolver import AlsSolver
from .AlsInitializer import AlsInitializer


class Als:
    
    """
        Top-level class for executing ALS algorithm. Calls AlsInitializer for setup and AlsSolver for executing optimization. Entry-point method is fit().
        
    """
    
    
    def __init__(self):
        self.__R = None
        self.__user_embedding = None
        self.__item_embedding = None
        self.__mse = None


    @property
    def R(self):
        return self.__R
    

    @property
    def user_embedding(self):
        return self.__user_embedding


    @property
    def item_embedding(self):
        return self.__item_embedding

    @property
    def mse(self):
        return self.__mse


    @staticmethod
    def get_number_users(data:pd.DataFrame, user_col:str):
        return data[user_col].nunique()
    

    @staticmethod
    def get_number_items(data:pd.DataFrame, item_col:str):
        return data[item_col].nunique()


    


    def __initialize_als(self, data:pd.DataFrame, user_col:str, item_col:str, value_col:str, dim:int, interaction_type:str, lambda_u:int, lambda_i:int,
                         initial_U:np.array,initial_V:np.array, njobs:int):
                
        # Applies verifications to input values
        AlsInitializer.assert_input_values(
            data=data,
            user_col=user_col,
            item_col=item_col,
            value_col=value_col,
            allowed_interactions=['n_interactions','has_interacted','rating'],
            input_interaction=interaction_type,
            initial_U=initial_U,
            initial_V=initial_V,
            dim=dim,
            lambda_u=lambda_u,
            lambda_i=lambda_i,
            njobs=njobs
        )

        # Gets user-item interaction
        self.__R, user_map, item_map = AlsInitializer.create_user_item_correspondence(data=data,                                                                                    user_col=user_col,
                                                                                      item_col=item_col,
                                                                                      value_col=value_col,
                                                                                      interaction_type=interaction_type)

        # Initializes embeddings
        U, V = AlsInitializer.initialize_embeddings(
            n_users=self.__R['user'].nunique(),
            n_items=self.__R['item'].nunique(),
            dim=dim,
            initial_U=initial_U,
            initial_V=initial_V
        )

        return U, V, user_map, item_map


    def __run_als(self, ui_corr:pd.DataFrame, U:np.array, V:np.array, dim:int,lambda_u:int, lambda_i:int, tol:float, njobs:int):
        
        # Creates solver and starts ALS
        solver = AlsSolver(
            R=ui_corr,
            U=U,
            V=V,
            lambda_u=lambda_u,
            lambda_i=lambda_i,
            tol=tol,
            njobs=njobs

        )
        U, V, mse = solver.solve()

        return U, V, mse

    
    def __format_attributes(self, U:np.array, V:np.array, user_map:dict, item_map:dict, user_col:str, item_col:str):
        
        # Inverse mappings
        inv_user_map = {v: k for k, v in user_map.items()}
        inv_item_map = {v: k for k, v in item_map.items()}

        # Saves normalized embeddings
        self.__user_embedding = {inv_user_map[i]:U[i]/np.linalg.norm(U[i])\
                                                    for i in range(U.shape[0])}
        self.__item_embedding = {inv_item_map[j]:V[j]/np.linalg.norm(V[j])\
                                                    for j in range(V.shape[0])}

        # Formats user-item interaction matrix
        self.__R['user'] = self.__R['user'].map(inv_user_map)
        self.__R['item'] = self.__R['item'].map(inv_item_map)
        self.__R = self.__R.drop(columns=['interaction_predict','err'])\
                           .rename(columns={'user':user_col,'item':item_col})


    def __save_embeddings(self, U:np.array, V:np.array, export_dir:str):

            if export_dir[-1]!='/':
                export_dir = export_dir+'/'

            with open(export_dir+'user_embedding.pkl','wb') as file:
                pickle.dump(self.__user_embedding,file)
            with open(export_dir+'item_embedding.pkl','wb') as file:
                pickle.dump(self.__item_embedding,file)


    def fit(
        self,
        data:pd.DataFrame, 
        user_col:str, 
        item_col:str, 
        value_col:str=None, 
        dim:int=10, 
        interaction_type:str='n_interactions', 
        lambda_u:float=0, 
        lambda_i:float=0,
        initial_U:np.array=None, 
        initial_V:np.array=None, 
        tol:float=0.1, 
        njobs:int=1, 
        export_dir:str=None
    ):
        
        """
            Entry point for executing the ALS algorithm.
            INPUTS:
            
                - data (pandas DataFrame): user-item interaction matrix in the form [user_col, item_col, interaction_col].
                
                - user_col (str): column with user ids in data.
                
                - user_col (str): column with item ids in data.
                
                - value_col (str): column with interaction values. 
                
                - dim (int): embedding dimension. Returned user and item embeddings will be 1-D arrays 
                of size dim.
                
                - interaction_type (str): type of interaction between users and items upon which to create embeddings. This is not 
                necessarily the kind of interaction portrayed in <data>, but how the algorithm will process the input user-item
                interaction matrix. Currently supports three kinds of interaction_types:
                    - n_interactions: algorithm will count how many user-item interaction events are present in <data>.
                    - has_interacted: algorithm will check whether a user-item interaction is present in <data> or not.
                    - rating: algorithm will consider a score a user gives an item. For this mode of operation, user-item
                    interaction matrix <data> must already contain ratings in <value_col>.
                    
                -lambda_u (float or double): regularization coefficient for user embeddings.
                -lambda_i (float or double): regularization coefficient for item embeddings.
                
                -initial_U (array-type): initial embedding for users. Must be a 1-D array with <dim> elements. If not specified, will be initialized
                internally with random values from standard normal distribution.
                -initial_i (array-type): initial embedding for items. Must be a 1-D array with <dim> elements. If not specified, will be initialized
                internally with random values from standard normal distribution.
                
                - tol (float): error tolerance for concluding optimization. For example, if tol=0.1, algorithm will execute until an iteration reduces mse
                in less than 10% wrt to previous iteration mse.
                
                - njobs (int): number of parallel jobs to execute optimization with.
                
                - export_dir (str): directory to which user and item embeddings will be exported. If not specified, will not export.
        
        """
        
        # Asserts inputs and initializes vars
        U, V, user_map, item_map = self.__initialize_als(
            data=data,
            user_col=user_col,
            item_col=item_col,
            value_col=value_col,
            dim=dim,
            interaction_type=interaction_type,
            lambda_u=lambda_u,
            lambda_i=lambda_i,
            initial_U=initial_U,
            initial_V=initial_V,
            njobs=njobs
        )


        # Runs ALS algorithm
        U, V, mse = self.__run_als(
            ui_corr=self.__R,
            U=U,
            V=V,
            dim=dim,
            lambda_u=lambda_u,
            lambda_i=lambda_i,
            tol=tol,
            njobs=njobs
        )
        self.__mse = mse

        # Formats attributes (embeddings and user-item interaction)
        self.__format_attributes(
            U=U,
            V=V,
            user_map=user_map,
            item_map=item_map,
            user_col=user_col,
            item_col=item_col
        )
        
        # If export directory is provided, save embeddings to directory
        if export_dir is not None:
            self.__save_embeddings(
                U=U,
                V=V,
                export_dir=export_dir
            )


            
