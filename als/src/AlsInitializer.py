import pandas as pd
import numpy as np

class AlsInitializer:

    def __init__(self):
        pass
    
    @staticmethod
    def assert_input_values(data:pd.DataFrame, user_col:str, item_col:str, value_col:str, allowed_interactions:np.array, input_interaction:str, lambda_u:float, lambda_i:float, 
                            initial_U:np.array, initial_V:np.array, dim:int, njobs:int):
        
        # Checks if specified interaction type is allowed
        if input_interaction not in allowed_interactions:
            raise AssertionError('Interaction type \'{}\' not allowed. Possible options are: {}'\
                        .format(input_interaction,
                                ','.join(['\''+x+'\'' for x in allowed_interactions])))

        # If interaction type is ratings, column in which ratings are given must be specified
        if input_interaction=='rating' and value_col is None:
            raise AssertionError('Interaction type \'rating\' demands specifying column which stores user ratings for items.')

        # If initial_U or initial_V are specified, checks if embedding dimensions match the one specified in dim and if
        # number of unique users and items are the same as in data
        if initial_U is not None:
            if initial_U.shape[1] != dim:
                raise AssertionError('Given initial U array has shape {}, which is incompatible with embedding dimension given: {}'\
                                     .format(initial_U.shape,dim))
            if initial_U.shape[0] != data[user_col].nunique():
                raise AssertionError('Data specified has {} unique users, which is incompatible with embedding dimension given: {}'\
                                     .format(data[user_col].nunique(), initial_U.shape))


        if initial_V is not None:
            if initial_V.shape[1] != dim:
                raise AssertionError('Given initial V array has shape {}, which is incompatible with embedding dimension given: {}'\
                                     .format(initial_V.shape,dim))
            if initial_V.shape[0] != data[item_col].nunique():
                raise AssertionError('Data specified has {} unique items, which is incompatible with embedding dimension given: {}'\
                                     .format(data[item_col].nunique(), initial_V.shape))

        # Check if regularization params are not negative
        if (lambda_u < 0) or (lambda_i <0):
            raise AssertionError('Regularization weight cannot be negative.')

        # Checks if user_col and item_col are columns of data
        if user_col not in data.columns:
            raise AssertionError('User column \'{}\' not in specified data.'.format(user_col))
        if item_col not in data.columns:
            raise AssertionError('Item column \'{}\' not in specified data.'.format(item_col))

        


    @classmethod
    def __create_id_mappings(cls, user_item_corr:pd.DataFrame, user_col:str, item_col:str, unique_users:np.array, unique_items:np.array):
        
        # Creates user and item mappings
        user_map = dict(zip(unique_users,range(len(unique_users))))
        item_map = dict(zip(unique_items,range(len(unique_items))))

        # Applies mappings
        user_item_corr[user_col] = user_item_corr[user_col].map(user_map)
        user_item_corr[item_col] = user_item_corr[item_col].map(item_map)

        return user_item_corr, user_map, item_map


    @classmethod
    def create_user_item_correspondence(cls, data:pd.DataFrame, user_col:str, item_col:str, value_col:str, interaction_type:str):
        """
        
        """
        # 
        if interaction_type=='rating':
            user_item_corr = data.copy(deep=True)
            del data

        elif interaction_type in ['n_interactions','has_interacted']:
            # Creates aggregation function according to interaction_type
            if interaction_type=='n_interactions':
                aggregation_fn = lambda x:x.shape[0]
            elif interaction_type=='has_interacted':
                aggregation_fn = lambda x:1

            # Aggregates
            data['agg_col'] = np.nan
            user_item_corr = data.groupby(by=[user_col,item_col],as_index=False)\
                                .agg({'agg_col':aggregation_fn})\
                                .rename(columns={'agg_col':'interaction'})\
                                .reset_index(drop=True)
            del data

        # Creates then applies user and item mappings
        user_item_corr, user_map, item_map = cls.__create_id_mappings(
            user_item_corr=user_item_corr,
            user_col=user_col,
            item_col=item_col,
            unique_users=user_item_corr[user_col].unique(),
            unique_items=user_item_corr[item_col].unique()
        )
         
        # Standardizes user-item interaction matrix's column names for later steps
        if interaction_type=='rating':
            user_item_corr=user_item_corr.rename(
                columns={user_col:'user',
                         item_col:'item',
                         value_col:'interaction'}
            )
        elif interaction_type in ['n_interactions','has_interacted']:
            user_item_corr = user_item_corr.rename(columns={user_col:'user',
                                                            item_col:'item'})
        return user_item_corr, user_map, item_map


    @staticmethod
    def initialize_embeddings(n_users:int, n_items:int, dim:int, initial_U:np.array, initial_V:np.array):
        
        if initial_U is None:
            U = np.random.normal(
                loc=0,
                scale=1,
                size=(n_users,dim)
            )
        else:
            U = initial_U

        if initial_V is None:
            V = np.random.normal(
                loc=0,
                scale=1,
                size=(n_items,dim)
            )
        else:
            V = initial_V

        return U, V