import simpy
import globalvar

class Task():
  
    def __init__(self, id_, env):
        self.id_ = id_
        self.env = env
        self.start = env.now
        self.end = 0
        self.completionEvent = simpy.events.Event(env)


    def sigTaskComplete(self):

        if self.completionEvent != None:
            self.completionEvent.succeed()
            print("%s was completed at time %d" %(self.getID(), self.env.now))
            print("It took " + str(self.env.now-self.start) + " to complete.")


            #monitor latency
            if self.env.now > globalvar.lengthOfInterval*globalvar.l:
                globalvar.timeToComplete.append(globalvar.time)
                globalvar.time = []
                globalvar.l = globalvar.l + 1
            self.end = self.env.now
            globalvar.time.append(self.end - self.start)

    def getID(self):
        return self.id_
