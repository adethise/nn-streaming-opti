import numpy as np
import simpy
import node
import workload
import globalvar



def runExperiment():

    env = simpy.Environment()


    #create topology
    bolts = []
    spouts = []

    for i in range(globalvar.numberOfNodes):
    
        if i < globalvar.numberOfSpouts:
            s = node.Node(env, 'Spout %d' % i, i, True, False)
            spouts.append(s)
            print(s.getName() + " created")
            
        elif i < (globalvar.numberOfSpouts+globalvar.numberOfEnds):
            b = node.Node(env, 'Bolt %d' % (i-globalvar.numberOfSpouts),i, False, False)
            bolts.append(b)
            print(b.getName() + " created")
        else:
            b = node.Node(env, 'Bolt %d' % (i-globalvar.numberOfSpouts),i, False, True)
            bolts.append(b)
            print(b.getName() + " created")

    #topology

    spouts[0].addNeighbor(bolts[0])
    spouts[0].addNeighbor(bolts[1])
    spouts[0].addNeighbor(bolts[2])

    spouts[1].addNeighbor(bolts[0])
    spouts[1].addNeighbor(bolts[1])
    spouts[1].addNeighbor(bolts[2])

    spouts[2].addNeighbor(bolts[0])
    spouts[2].addNeighbor(bolts[1])
    spouts[2].addNeighbor(bolts[2])

    bolts[0].addNeighbor(bolts[3])
    bolts[0].addNeighbor(bolts[4])
    bolts[0].addNeighbor(bolts[5])


    bolts[1].addNeighbor(bolts[3])
    bolts[1].addNeighbor(bolts[4])
    bolts[1].addNeighbor(bolts[5])


    bolts[2].addNeighbor(bolts[3])
    bolts[2].addNeighbor(bolts[4])
    bolts[2].addNeighbor(bolts[5])

    #create workload

    w = workload.Workload(env, "1", 3)

    w.addSpout(spouts[0])
    w.addSpout(spouts[1])
    w.addSpout(spouts[2])


    env.run(until=globalvar.duration)
    
    print("-----")
    print("-----")
    print("Monitoring")
    print("-----")
    print('Throughput: ', globalvar.throughput)
    print("-----")
    print('Latency:')
    print('    0.90 quantile')
    for i in range(globalvar.numberOfIntervals-1):
        if(globalvar.timeToComplete[i] != []):
            print("    Interval " + str(i+1) + ":" + str(np.percentile(globalvar.timeToComplete[i],90)))
        else:
            print("    Interval " + str(i+1) + ":" + str(globalvar.timeToComplete[i]))
    print("    -----")
    print('    0.95 quantile')
    for i in range(globalvar.numberOfIntervals-1):
        if(globalvar.timeToComplete[i] != []):
            print("    Interval " + str(i+1) + ":" + str(np.percentile(globalvar.timeToComplete[i],95)))
        else:
            print("    Interval " + str(i+1) + ":" + str(globalvar.timeToComplete[i]))
    print("    -----")
    print('    0.99 quantile')
    for i in range(globalvar.numberOfIntervals-1):
        if(globalvar.timeToComplete[i] != []):
            print("    Interval " + str(i+1) + ":" + str(np.percentile(globalvar.timeToComplete[i],99)))
        else:
            print("    Interval " + str(i+1) + ":" + str(globalvar.timeToComplete[i]))
    print('    0.5 quantile')
    for i in range(globalvar.numberOfIntervals-1):
        if(globalvar.timeToComplete[i] != []):
            print("    Interval " + str(i+1) + ":" + str(np.percentile(globalvar.timeToComplete[i],50)))
        else:
            print("    Interval " + str(i+1) + ":" + str(globalvar.timeToComplete[i]))
    print("    -----")
    print("-----")
    print("CPU utilization (per node): ")
    for i in range(globalvar.numberOfIntervals-1):
        print("Interval "+ str(i+1) + ":" + str(globalvar.cpu[i]))
    print("-----")
    print("Number of arrivals (per node): ")
    for i in range(globalvar.numberOfIntervals-1):
        print("Inteval "+ str(i+1) + ":" + str(globalvar.numberOfArrivals[i]))

    print('After '+ str(globalvar.i-1)+ ' intervals')


    #print(np.sort(globalvar.timeToComplete[0])) 

if __name__ == '__main__':
     runExperiment()


                


