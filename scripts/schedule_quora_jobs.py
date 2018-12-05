import os
import pdb
import subprocess as sp

OUTPUT_ROOT='/scratch/cluster/pkar/lin393s-project/runs/sse_quora_only_glove'
SCRIPT_ROOT='/scratch/cluster/pkar/lin393s-project/scripts/'

mapping_dict = {
    # Condor Scheduling Parameters
    '__EMAILID__': 'pkar@cs.utexas.edu',
    '__PROJECT__': 'INSTRUCTIONAL',
    # Script parameters
    '__JOBNAME__': ['sse_quora_only_lr_1e-3', 'sse_quora_only_lr_3e-4', 'sse_quora_only_lr_1e-4'],
    # Algorithm hyperparameters
    '__CODE_ROOT__': '/scratch/cluster/pkar/lin393s-project',
    '__MODE__': 'train_quora',
    '__DATA_DIR__': '/scratch/cluster/pkar/lin393s-project/data/',
    '__NWORKERS__': '4',
    '__BSIZE__': '48',
    '__SHUFFLE__': 'True',
    '__GLOVE_EMB_FILE__': '/scratch/cluster/pkar/lin393s-project/data/glove.6B/glove.6B.300d.txt',
    '__MAXLEN__': '60',
    '__N_CANDIDATE_RESP__': '10',
    '__ARCH__': 'sse_multitask',
    '__HIDDEN_SIZE__': '300',
    '__NUM_LAYERS__': '1',
    '__BIDIRECTIONAL__': 'False',
    '__PRETRAINED_EMB__': 'True',
    '__DROPOUT_P__': '0.2',
    '__OPTIM__': 'adam',
    '__LR__': ['1e-3', '3e-4', '1e-4'],
    '__WD__': '4e-5',
    '__MOMENTUM__': '0.9',
    '__EPOCHS__': '40',
    '__MAX_NORM__': '10',
    '__LR_DECAY_STEP__': '50',
    '__LR_DECAY_GAMMA__': '0.1',
    '__START_EPOCH__': '0',
    '__LOG_ITER__': '10',
    '__RESUME__': 'False',
    '__SEED__': '123',
    }

# Figure out number of jobs to run
num_jobs = 1
for key, value in mapping_dict.items():
    if type(value) == type([]):
        if num_jobs == 1:
            num_jobs = len(value)
        else:
            assert(num_jobs == len(value))

for idx in range(num_jobs):
    job_name = mapping_dict['__JOBNAME__'][idx]
    mapping_dict['__LOGNAME__'] = os.path.join(OUTPUT_ROOT, job_name)
    if os.path.isdir(mapping_dict['__LOGNAME__']):
        print ('Skipping job ', mapping_dict['__LOGNAME__'], ' directory exists')
        continue

    mapping_dict['__LOG_DIR__'] = mapping_dict['__LOGNAME__']
    mapping_dict['__SAVE_PATH__'] = mapping_dict['__LOGNAME__']
    sp.call('mkdir %s'%(mapping_dict['__LOGNAME__']), shell=True)
    condor_script_path = os.path.join(mapping_dict['__SAVE_PATH__'], 'condor_script.sh')
    script_path = os.path.join(mapping_dict['__SAVE_PATH__'], 'run_script.sh')
    sp.call('cp %s %s'%(os.path.join(SCRIPT_ROOT, 'condor_script_proto.sh'), condor_script_path), shell=True)
    sp.call('cp %s %s'%(os.path.join(SCRIPT_ROOT, 'run_proto.sh'), script_path), shell=True)
    for key, value in mapping_dict.items():
        if type(value) == type([]):
            sp.call('sed -i "s#%s#%s#g" %s'%(key, value[idx], script_path), shell=True)
            sp.call('sed -i "s#%s#%s#g" %s'%(key, value[idx], condor_script_path), shell=True)
        else:
            sp.call('sed -i "s#%s#%s#g" %s'%(key, value, script_path), shell=True)
            sp.call('sed -i "s#%s#%s#g" %s'%(key, value, condor_script_path), shell=True)

    sp.call('condor_submit %s'%(condor_script_path), shell=True)
