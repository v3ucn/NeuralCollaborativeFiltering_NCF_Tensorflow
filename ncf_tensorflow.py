import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# set pandas to show all columns without truncation and line breaks
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

# data = np.loadtxt('data/test-data.csv', delimiter=',', dtype=int, skiprows=1,)
data = pd.read_csv('data/test-data.csv')
print(data)

# reset the column.index to be numeric
user_index = data[data.columns[0]]
video_index = data.columns
data = data.reset_index(drop=True)
data[data.columns[0]] = data.index.astype('int')
# print(data)
# print(data)
scaler = 10

# data = pd.DataFrame(data.to_numpy(), index=range(0,len(user_index)), columns=range(0,len(video_index)))
df_long = pd.melt(data, id_vars=[data.columns[0]], 
                  ignore_index=True, 
                  var_name='video_id', 
                  value_name='rate').dropna()
df_long.columns = ['user_id', 'video_id', 'rating']
df_long['rating'] = df_long['rating'] / scaler
# replace the user_id to user by match user_index
df_long['user_id'] = df_long['user_id'].apply(lambda x: user_index[x])
# data = df_long.to_numpy()

#print(df_long)

dataset = df_long
# Encode the user and movie IDs
user_encoder = LabelEncoder()
video_encoder = LabelEncoder()
dataset['user_id'] = user_encoder.fit_transform(dataset['user_id'])
dataset['video_id'] = video_encoder.fit_transform(dataset['video_id'])

# Split the dataset into train and test sets
# train, test = train_test_split(dataset, test_size=0.2, random_state=42)
train = dataset

# Model hyperparameters
num_users = len(dataset['user_id'].unique())
num_countries = len(dataset['video_id'].unique())


embedding_dim = 64

# Create the NCF model
inputs_user = tf.keras.layers.Input(shape=(1,))
inputs_video = tf.keras.layers.Input(shape=(1,))
embedding_user = tf.keras.layers.Embedding(num_users, embedding_dim)(inputs_user)
embedding_video = tf.keras.layers.Embedding(num_countries, embedding_dim)(inputs_video)

# Merge the embeddings using concatenation, you can also try other merging methods like dot product or multiplication
merged = tf.keras.layers.Concatenate()([embedding_user, embedding_video])
merged = tf.keras.layers.Flatten()(merged)

# Add fully connected layers
dense = tf.keras.layers.Dense(64, activation='relu')(merged)
dense = tf.keras.layers.Dense(32, activation='relu')(dense)
output = tf.keras.layers.Dense(1, activation='sigmoid')(dense)

# Compile the model
model = tf.keras.Model(inputs=[inputs_user, inputs_video], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(
    [train['user_id'].values, train['video_id'].values],
    train['rating'].values,
    batch_size=64,
    epochs=100,
    verbose=0,
    # validation_split=0.1,
)

result_df = {}
for user_i in range(1, 10):
  user = f'User{user_i}'
  result_df[user] = {}
  for video_i in range(1, 7):    
    video = f'Video {video_i}'
    pred_user_id = user_encoder.transform([user])
    pred_video_id = video_encoder.transform([video])
    result = model.predict(x=[pred_user_id, pred_video_id], verbose=0)
    result_df[user][video] = result[0][0]
result_df = pd.DataFrame(result_df).T
result_df *= scaler

print(result_df)

