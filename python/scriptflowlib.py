"""
    This implements the library used by scripflow.py. This is very preliminary but works for the
    purpose of this replication package.

    See the latest version at https://github.com/tlamadon/scriptflow
"""


import subprocess
import os
import time
#import toml
import asyncio
import tempfile

from rich.console import Console
from time import sleep
from os.path import exists
from tinydb import TinyDB, Query
from datetime import datetime

console = Console()

from rich.console import Console
from time import sleep

"""

note: 
 - use input and output for dependencies and generated stuff
 - use set_args and set_return for input parameters and return values that should be collected

"""
class Task:
    cmd =""
    uid=""
    mem="32Gb"
    ncore="10"
    retry=0

    def __init__(self, cmd):
        self.fut = asyncio.get_event_loop().create_future()
        self.running = False
        self.task = None
        self.cmd = cmd
        self.deps = []
        self.output_file = ""
        self.quiet = True

    def __await__(self):
        if self.task is None:
            self.start()
        return self.fut.__await__()

    def start(self):
        self.running = True
        get_main_maestro().add(self)
        return(self)

    def result(self, return_file):
        self.return_file = return_file    
        return self  

    def output(self, output_file):
        self.output_file = output_file    
        return self     

    def retry(self, retry):
        self.retry = retry    
        return self     

    def input(self, input_file):
        # check that the input file actually exist and extract its values, too soon?
        self.input_file = input_file  
        return self  

    def add_deps(self, deps):

        if isinstance(deps, str):
            self.deps.append(deps)
            return(self)

        if not hasattr(deps, "__iter__"): #returns True if type of iterable - same problem with strings
            deps = list(deps)
        
        for dep in deps:
            self.deps.append(dep)

        return(self)

    def uid(self,uid):
        self.uid = uid
        return self

    def get_command(self):
        return self.cmd

class CommandRunner:

    def __init__(self,max_proc):
        self.max_proc = max_proc
        self.queue = []
        self.processes = {}
        self.finished = {}
        self.console = Console()
        self.done = 0
        self.history = TinyDB('sf.json')
        pass

    def add(self,cmd):
        self.queue.append(cmd)

    def log(self,str):
        console.log(str)

    async def loop(self):

        with console.status("Running ...") as status:
            while True:

                # check if we can add a new process
                if (len(self.queue)>0) & (len(self.processes) < self.max_proc):
                    
                    j = self.queue[0]
                    self.queue.remove(j)

                    # checking if the task needs to be redone
                    # to be done   
                    if os.path.exists(j.output_file):
                        # console.log("checking {}".format(j.output_file))
                        output_time = os.path.getmtime(j.output_file)  

                        UP_TO_DATE = True
                        for f in j.deps:
                            if os.path.getmtime(f) > output_time:    
                                console.log("{} input {} more recent than output".format(j.uid, f))
                                UP_TO_DATE = False
                                break 
                        
                        if UP_TO_DATE:
                            console.log(f"up to date, skipping [red]{j.uid}[/red]")
                            #self.queue.remove(j)
                            j.fut.set_result(j)
                            continue                                                     

                    console.log(f"adding [red]{j.uid}[/red]")
                    console.log("cmd: {}".format( " ".join(j.get_command() ) ))

                    if j.quiet:
                        subp = subprocess.Popen(j.get_command(),
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT)
                    else:
                        subp = subprocess.Popen(j.get_command())
                            #stdout=subprocess.DEVNULL,
                            #stderr=subprocess.STDOUT)


                    self.processes[j.uid] = {
                        "proc" : subp, 
                        "job": j,  
                        'start_time': datetime.now().strftime("%d/%m/%Y %H:%M:%S")}     
                                   
                    #self.queue.remove(j)
                    status.update(f"running [green]queued:{len(self.queue)}[/green] [yellow]running:{len(self.processes)}[/yellow] [purple]done:{self.done}[/purple] ...")
                
                to_remove = []
                for (k,p) in self.processes.items():
                    poll_val = p["proc"].poll()
                    if poll_val is not None:
                        console.log("job {} done r={}".format(p["job"].uid, poll_val))   
                        
                        to_remove.append(k)
                        self.finished[p["job"].uid] = True
                        self.done = self.done +1
                        p["job"].fut.set_result(p["job"])

                        # append job to history 
                        tj = Query()
                        if len(self.history.search(tj.uid ==  p["job"].uid))>0:
                            self.history.update({
                                'deps' : p["job"].deps,
                                'output' : p["job"].output_file,
                                'cmd':p["job"].get_command(),
                                'start_time':p['start_time'],
                                'end_time': datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                            },tj.uid ==  p["job"].uid)
                        else:
                            self.history.insert({
                                'uid': p["job"].uid, 
                                'deps' : p["job"].deps,
                                'output' : p["job"].output_file,
                                'cmd':p["job"].get_command(),
                                'start_time':p['start_time'],
                                'end_time': datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                            })
                            
                for k in to_remove:
                    del self.processes[k]                 
                    status.update(f"running [green]queued:{len(self.queue)}[/green] [yellow]running:{len(self.processes)}[/yellow] [purple]done:{self.done}[/purple] ...")

                await asyncio.sleep(0.1)

class HpcRunner:

    script_template = """
    #PBS -N {name}
    #PBS -j oe
    #PBS -V
    #PBS -l procs={procs},mem={mem}
    #PBS -l walltime=12:00:00

    module load julia/1.6.1
    cd {wd}

    {cmd}
    """

    def __init__(self,max_proc):
        self.max_proc = max_proc
        self.queue = []
        self.processes = {}
        self.finished = {}
        self.console = Console()
        self.done = 0
        self.failed = 0
        self.job_params = {'procs':1, 'mem' : "16Gb", 'name':'psub'}
        pass

    def add(self,job):
        self.queue.append(job)

    def log(self,str):
        console.log(str)

    async def loop(self):
        with console.status("Running ...") as status:
            while True:

                # check if we can add a new process
                if (len(self.queue)>0) & (len(self.processes) < self.max_proc):
                    
                    j = self.queue[0]

                    # checking if the task needs to be redone
                    # to be done   
                    if os.path.exists(j.output_file):
                        # console.log("checking {}".format(j.output_file))
                        output_time = os.path.getmtime(j.output_file)  

                        UP_TO_DATE = True
                        for f in j.deps:
                            if os.path.getmtime(f) > output_time:    
                                console.log("{} input {} more recent than output".format(j.uid, f))
                                UP_TO_DATE = False
                                break 
                        
                        if UP_TO_DATE:
                            console.log(f"up to date, skipping [red]{j.uid}[/red]")
                            self.queue.remove(j)
                            j.fut.set_result(j)
                            continue       

                    console.log(f"adding [blue]{j.uid}[/blue]")
                    console.log("cmd: {}".format( " ".join(j.get_command() ) ))

                    # create the script
                    script_content = self.script_template.format(
                        name = "sf-{}".format(j.uid),
                        mem = j.mem,
                        procs = j.ncore,
                        wd = os.getcwd(), 
                        cmd = " ".join(j.get_command()) )

                    tmp = tempfile.NamedTemporaryFile(delete=False)
                    tmp_script_filename = tmp.name
                    tmp.write(script_content.encode())
                    tmp.close()

                    command = ["qsub", tmp_script_filename]
                    try:
                        output = subprocess.check_output(command).decode()
                        JOB_ID = output.replace("\n","")
                        
                        self.processes[j.uid] = {"JOB_ID" : JOB_ID, "job": j, "status":"S"}          
                        self.queue.remove(j)
                        status.update(f"running [green]queued:{len(self.queue)}[/green] [yellow]running:{len(self.processes)}[/yellow] [purple]done:{self.done}[/purple] [purple]failed:{self.failed}[/purple] ...")

                    except subprocess.CalledProcessError as e:
                        console.log("issue with qsub, giving it a short break...")
                        status.update(f"running [green]queued:{len(self.queue)}[/green] [yellow]running:{len(self.processes)}[/yellow] [purple]done:{self.done}[/purple] [purple]failed:{self.failed}[/purple] ...")
                        await asyncio.sleep(2)

                    continue
                
                # checking job status
                output = subprocess.check_output("qstat").decode()
                lines = output.split("\n") 
                job_status = {}
                for l in lines[2:]:                    
                    vals = l.split()
                    if len(vals)<3:
                        continue
                    job_status[vals[0]] = {'status':vals[4]}

                to_remove = []
                for (k,p) in self.processes.items():

                    if p["JOB_ID"] not in job_status.keys():
                        continue

                    if job_status[p["JOB_ID"]]['status'] == 'C':
                        console.log("job {} finished".format(p["job"].uid))   

                        # checking that output file exists
                        if not exists(p["job"].return_file):

                            if  p["job"].retry>0:
                                console.log("[red]output file missing for {}, retrying... [/red]".format(p["job"].uid))  
                                p["job"].retry = p["job"].retry -1
                                self.add(p["job"])
                            else:
                                console.log("[red]output file missing for {}, failing! [/red]".format(p["job"].uid))  
                                self.finished[p["job"].uid] = True
                                self.failed = self.failed + 1
                                p["job"].fut.set_result(p["job"])

                        else:
                            self.finished[p["job"].uid] = True
                            self.done = self.done +1
                            p["job"].fut.set_result(p["job"])

                        to_remove.append(k)

                for k in to_remove:
                    del self.processes[k]                 
                    status.update(f"running [green]queued:{len(self.queue)}[/green] [yellow]running:{len(self.processes)}[/yellow] [purple]done:{self.done}[/purple] [purple]failed:{self.failed}[/purple] ...")

                await asyncio.sleep(2)


# async def main():
#     cr = CommandRunner(3)    
#     cr.log("starting loop")
#     loop = asyncio.create_task(cr.loop())

#     # cr.add(Task(["python", "-c", "import time; time.sleep(3); print('done')"]).uid("task1"))
#     # cr.add(Task(["python", "-c", "import time; time.sleep(3); print('done')"]).uid("task2"))

#     t1 = cr.createTask(Task(["python", "-c", "import time; time.sleep(5); print('done')"]).uid("task1"))
#     t2 = cr.createTask(Task(["python", "-c", "import time; time.sleep(5); print('done')"]).uid("task2"))

#     await asyncio.gather(t1,t2)

# asyncio.run(main())


# creating the main executor!
maestro = None

def set_main_maestro(cr):
    global maestro
    maestro = cr

def get_main_maestro():
    return maestro
