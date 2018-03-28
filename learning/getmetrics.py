import json
from stormmetricscollector import StormMetrics

metrics = StormMetrics("8080")
# If you have multiple topologies running then you will have to select the
# topology by the index before calling the function to get metrics
#for example like
#metrics.setTopology(0)

print('JSON:\n', metrics.getJson())
print('Bolts:\n', metrics.getBolts())
print('Capacity:\n', metrics.getAllCapacity())
print('BoltStats:\n', metrics.getAllBoltStats())
print('SpoutStats:\n', metrics.getAllSpoutStats())
