# data_preparation.py

import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler

class DataPreparation:
    def __init__(self, processed_df, user_features=None):
        self.processed_df = processed_df
        self.user_features = user_features
        self.data = self.prepare_data()

    def prepare_data(self):
        data = HeteroData()

        # Assign 'tweet' node features
        data['tweet'].x = torch.tensor(self.processed_df['embedding'].tolist(), dtype=torch.float)

        # Assign 'user' node features
        if self.user_features is not None:
            data['user'].x = self.user_features
        else:
            # Create user features from available columns
            user_features_df = self.processed_df.groupby('user_id').agg({
                'user_followers_count': 'mean',
                'user_friends_count': 'mean'
            }).reset_index()

            # Normalize the features
            scaler = StandardScaler()
            user_features_df[['user_followers_count', 'user_friends_count']] = scaler.fit_transform(
                user_features_df[['user_followers_count', 'user_friends_count']]
            )

            # Convert to torch tensor
            user_features_tensor = torch.tensor(
                user_features_df[['user_followers_count', 'user_friends_count']].values, 
                dtype=torch.float
            )

            data['user'].x = user_features_tensor

        # Define edges
        # Example: 'user' posts 'tweet'
        user_ids = self.processed_df['user_id'].unique()
        tweet_ids = self.processed_df['tweet_id'].unique()

        user_id_to_idx = {id: idx for idx, id in enumerate(user_ids)}
        tweet_id_to_idx = {id: idx for idx, id in enumerate(tweet_ids)}

        source = self.processed_df['user_id'].map(user_id_to_idx).tolist()
        target = self.processed_df['tweet_id'].map(tweet_id_to_idx).tolist()

        data['user', 'posts', 'tweet'].edge_index = torch.tensor([source, target], dtype=torch.long)

        # Add other edge types as needed

        return data
