import dataStructures as _

features = {}

tweet = _.Tweet()
user = _.User()

f = _.CapitalizationFeature()
features[f.getKey()] = f.getValue(tweet)

f = _.AverageTweetLengthFeature()
features[f.getKey()] = f.getValue(user)
