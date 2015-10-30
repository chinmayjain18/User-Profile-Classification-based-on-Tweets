from dataStructures import *

# Data setup for example (because we don't have real tweets or users yet)
tweet = Tweet()
user = User()

# Add features to array
f_objects = []
f[0] = CapitalizationFeature(tweet)
f[1] = AverageTweetLengthFeature(user)

# Generate features dictionary from features
features = {}
for f in f_objects:
    features[f.getKey()] = f.getValue()
