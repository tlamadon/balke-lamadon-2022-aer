#!/usr/bin/python3

# simple script flow

from email import policy
import os
import asyncio
import click
import numpy as np
import shutil
import glob

import asyncio
from time import sleep
from pathlib import Path

import scriptflowlib as sf

@click.group()
def cli():
    pass

# ----------------------------- TASKS -----------------------

def task_cf_vdec_level(i):
    
    target = "build/cf/vdec-level-noise0-rep{}.json".format(i)
    t = sf.Task(["python", "-c", "import results as cf; cf.cf_simulate_level_var_dec({},{},False)".format("\"{}\"".format(target), i)])
    t.output(target).uid("cf-vdec-level-{}".format(i)).add_deps("res_main_model.pkl")
    return t

def task_cf_vdec_growth(i):
    target = "build/cf/vdec-growth-noise0-weight0-rep{}.json".format(i)
    t = sf.Task(["python", "-c", "import results as cf; cf.cf_vardec_growth_rep_one({},{},200,20000,False,False)".format(i, "\"{}\"".format(target), i)])
    t.output(target).uid("cf-vdec-growth-{}".format(i)).add_deps("res_main_model.pkl")
    return t

"""
Task that solves the model once taking a given file as input and 
generating an output file next to it
"""
def task_solve_model(input_file):

    file = Path(input_file)
    target = file.with_name( file.stem + '_moments.pkl')

    t = sf.Task(["python", "main_model_eval_once.py", "-p", str(file) , "-s", str(target) ])
    t.uid("model-once-{}".format(file.stem))
    t.output(str(target))
    t.add_deps(str(file))

    return(t)

def task_python(code):
    return sf.Task(["python", "-c", code])

# ----------------------------- FLOWS -----------------------
async def flow_model():

    # save the model
    t1 = sf.Task(["python", "-c", "import results as cf; cf.save_model_solution('../results/parameters_at_optimal.json')"])
    t1.output('res_main_model.pkl')
    t1.add_deps('../results/parameters_at_optimal.json')
    t1.uid("solve-model")
    await t1

    # save first best
    t2 = sf.Task(["python", "-c", "import results as cf; cf.save_model_solution_fb()"])
    t2.add_deps('res_main_model.pkl').output('res_main_model_fb.pkl')
    t2.uid("solve-first-best")
    await t2

    # save fit
    t2 = sf.Task(["python", "-c", "import results as cf; cf.save_model_fit()"])
    t2.add_deps('res_main_model.pkl').output('../figures/table1-moments.tex')
    t2.uid("get-model-fit")
    await t2

"""
Flow that creates the table for cross-section decompositions
"""
async def flow_cf_vdec_level():

    Path("build/cf").mkdir(parents=True, exist_ok=True)

    tasks = [task_cf_vdec_level(i) for i in range(20)]
    await asyncio.gather(*tasks)

    # generate the table
    table = sf.Task(["python", "-c", "import results; results.table_variance_decomposition_level()"])
    table.uid("cf-vdec-level").output("../figures/table3-vdec-cs.tex")
    table.add_deps(t.output_file for t in tasks)
    await table


"""
Flow that creates the table for growth decompositions
"""
async def flow_cf_vdec_growth():

    Path("build/cf").mkdir(parents=True, exist_ok=True)

    tasks = [task_cf_vdec_growth(i) for i in range(20)]
    await asyncio.gather(*tasks)

    # # generate the table
    table = sf.Task(["python", "-c", "import results; results.table_variance_decomposition_growth()"])
    table.uid("cf-vdec-growth").output("../figures/table4-vdec-growth.tex")
    table.add_deps(t.output_file for t in tasks)
    await table


async def flow_model_to_life():

    # # generate the table
    t1 = sf.Task(["python", "-c", "import results as cf; cf.cf_model_to_life(False)"])
    t1.uid("cf-model2life").output("../figures/figure3-ir-xshock.pdf")

    t2 = sf.Task(["python", "-c", "import results as cf; cf.cf_model_to_life(True)"])
    t2.uid("cf-model2life-fb").output("../figures/figurew5-ir-xshock-fb.pdf")

    await asyncio.gather(t1,t2)


async def flow_passthrough():
    t1 = sf.Task(["python", "-c", "import results as cf; cf.passthrough_analysis()"])
    t1.uid("cf-passthough").output('../figures/table5-passthrough.tex').input('"res_main_model.pkl"')
    await t1

async def flow_policy():

    # we start by creating all the models to solve
    policy_input1 = sf.Task(["python", "-c", "import results as cf; cf.cf_policy_gen_neutral()"])
    policy_input1.uid("policy-inputs-no-noise")
    policy_input1.output("build/policy/param_lhigh_direct_net0_nm.json")
    await policy_input1

    policy_input2 = sf.Task(["python", "-c", "import results as cf; cf.cf_policy_gen_neutral(True)"])
    policy_input2.uid("policy-inputs-with-noise")
    policy_input2.output("build/policy/param_lhigh_direct_net0.json")
    await policy_input2

    # get all input files that should be ran
    p = Path('build/policy')
    tasks = [task_solve_model(str(f)) for f in p.glob("param_*.json")]    
    await asyncio.gather(*tasks)

    policy_collect = sf.Task(["python", "-c", "import results as cf; cf.cf_policy_collect()"])
    policy_collect.uid("policy-collect")
    policy_collect.output("../figures/table6-policy.tex")
    await policy_collect

async def flow_slices():
    
    Path('build/slices').mkdir(parents=True, exist_ok=True)

    gen_inputs = sf.Task(["python", "-c", "import results as cf; cf.cf_slices_gen(25)"])
    gen_inputs.uid("slices-inputs")
    gen_inputs.output("build/slices/param_slice_zcorr_0.json")
    gen_inputs.quiet=False
    await gen_inputs

    p = Path('build/slices')
    tasks = [task_solve_model(str(f)) for f in p.glob("param_*.json")]    
    await asyncio.gather(*tasks)

    slice_collect = sf.Task(["python", "-c", "import results as cf; cf.cf_slices_collect()"])
    slice_collect.uid("slices-collect")
    slice_collect.output("build/slices.csv")
    slice_collect.quiet=False
    await slice_collect

async def flow_bootstrap():

    bs_gen = sf.Task(["python", "-c", "import results as cf; cf.cf_bootstrap_gen()"])
    bs_gen.uid("bootstrap-gen")
    bs_gen.output("build/bootstrap/param_pb_r0_p0.json")
    await bs_gen

    p = Path('build/bootstrap')
    tasks = [task_solve_model(str(f)) for f in p.glob("param_*.json")]    
    await asyncio.gather(*tasks)

    bs_collect = sf.Task(["python", "-c", "import results as cf; cf.cf_bootstrap_collect()"])
    bs_collect.uid("bootstrap-collect")
    bs_collect.output("../figures/table2-parameters.tex")
    await bs_collect
    
async def flow_sensitivity_figures():

    # we generate 3 tables
    t4 = task_python("import results as cf; cf.cf_sensitivity_measure()").uid("fig-sensitivity")
    await t4

async def flow_data_stats():

    # we generate 3 tables
    t4 = task_python("import results as cf; cf.table_stats()").uid("table-stats").output('../figures/tablew2-stats.tex')
    await t4

async def flow_pdfs():
    for f in Path("../figures").glob("*.tex"):
        latex_task = sf.Task(["python", "-c", "import results as cf; cf.generate_alone_pdf(\"{}\")".format(f)])
        latex_task.uid("latex_" + f.stem)
        await latex_task

async def flow_surrogate():
    sur_gen = sf.Task(["python", "-c", "import results as cf; cf.surrogate_gen(100)"])
    sur_gen.uid("surrogate-gen").output('build/surrogate/param_slice_efcostce_0.json')
    await sur_gen

    p = Path('build/surrogate')
    tasks = [task_solve_model(str(f)) for f in p.glob("param_*.json")]    
    await asyncio.gather(*tasks)

    bs_collect = sf.Task(["python", "-c", "import results as cf; cf.surrogate_collect()"])
    bs_collect.uid("bootstrap-collect")
    bs_collect.output("../figures/figurew1-surrogate.pdf")
    await bs_collect

# -----------------------------
#       OPTIMIZER 
# -----------------------------
async def flow_optimize():

    opt_gen = sf.Task(["python", "-c", "import results as cf; cf.optimize_gen(1,100)"])
    opt_gen.uid("opt-step1").output('build/opt/step1/param_slice_efcostce_0.json')
    await opt_gen

    for step in range(1,16):
        # we generate the set of parameters
        p = Path('build/opt/step{}'.format(step))
        tasks = [task_solve_model(str(f)) for f in p.glob("param_*.json")]    
        await asyncio.gather(*tasks)

        opt_gen = sf.Task(["python", "-c", "import results as cf; cf.optimize_gen({},100)".format(step+1)])
        opt_gen.uid("opt-step1").output('build/opt/step1/param_slice_efcostce_0.json')
        await opt_gen

async def flow_slices_and_bootstrap():
    # 1) we then compute slices, associated plots
    await flow_slices()

    # 2) we run bootstrap to get standard errors 
    await flow_bootstrap()

async def flow_fast():

    # we start by solving the model and the first best
    await flow_model()

    # we compute the variance decomposition tables and impulse response plots
    await asyncio.gather(
        flow_cf_vdec_level(),
        flow_cf_vdec_growth(),
        flow_model_to_life(),
        flow_policy(),
        flow_passthrough(),        
        flow_data_stats(),
        flow_surrogate())

    # reate pdfs for all latex files
    await flow_pdfs()

async def flow_part3():

    # 1) we start by solving the model and the first best
    await flow_model()

    # we compute the variance decomposition tables and impulse response plots
    await asyncio.gather(
        flow_cf_vdec_level(),
        flow_cf_vdec_growth(),
        flow_model_to_life(),
        flow_policy(),
        flow_passthrough(),
        flow_slices_and_bootstrap(),
        flow_data_stats())

    # 4 final tables and pdf output
    await flow_sensitivity_figures()

    # 5 create pdfs for all latex files
    await flow_pdfs()

async def flow_part2():
    await flow_optimize()

# flow aliases
async def flow_all():
    await asyncio.gather(
            flow_part2(),
            flow_part3()
    )

# flow aliases
async def flow_clean_all():

    shutil.rmtree('build')
    os.remove("../results/param_reoptimized.json")

    for f in glob.glob("../figures/*"):
        os.remove(f)

    await flow_all()

"""
    Main flow
"""
async def main(func):



    asyncio.create_task(sf.get_main_maestro().loop())
    os.makedirs('.sf', exist_ok=True)
    await func()      

@cli.command()
@click.argument('name')
@click.option('-n','--nodes', default=1)
def run(name,nodes):

    func_names = globals().keys()
    flows = [w.replace("flow_","") for w in func_names if w.startswith("flow_")]

    if name not in flows:
        print("Flow {} is not available, values ares: {}".format(name, ", ".join(flows)))
        return()

    cr = sf.CommandRunner(nodes)
    sf.set_main_maestro(cr)

    func = globals()["flow_{}".format(name)]
    asyncio.run(main(func))

if __name__ == '__main__':
    cli()
