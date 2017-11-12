
numberOfNodes = 9
numberOfSpouts = 3
numberOfEnds = 3

duration = 10000
lengthOfInterval = 1000
numberOfIntervals = round(duration/lengthOfInterval)


# an inverval will not be logged unless an event happens afterwards, 
# so last intevall will not be logged

# interval counter
i=1
j=1
k=1
l=1

# stores values per interval
t = 0
q = [0 for x in range(numberOfNodes)]
c = [0 for y in range(numberOfNodes)]
time = []


# stores all values

throughput = []
timeToComplete = []
numberOfArrivals = []
cpu = []





