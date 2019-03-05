from __future__ import print_function
import numpy as np
import pickle
import os
import time
import sys
import snep.configuration
import subprocess as sp
import multiprocessing as mp
from snep.tables.data import open_data_file

if hasattr(sp, 'run'):
    def run_local(c, print_cmd=True, print_out=True):
        stdout = ''
        if print_cmd:
            print('({}): {}'.format(snep.configuration.hostname, c))
        try:
            cp = sp.run(c, stdout=sp.PIPE, stderr=sp.PIPE, check=True, shell=True)
            stdout = cp.stdout.decode()
            if print_out:
                for l in stdout:
                    print(l, end='')
        except sp.CalledProcessError as e:
            print(e)
        return stdout
else:
    def run_local(c, print_cmd=True, print_out=True):
        if print_cmd:
            print('({}): {}'.format(snep.configuration.hostname, c))
        try:
            stdout = sp.check_output(c, shell=True)
            if print_out:
                for l in stdout:
                    print(l, end='')
        except sp.CalledProcessError as e:
            print(e)
            stdout = e.message
        return stdout


def run(JobInfo, ji_kwargs, resume=False, result_dir='experiments', delete_tmp=True, additional_files=None,
        max_tasks=None, max_task_time=None, mem_per_task=2, poll_interval=1., username=None, conda_env=None):
    """
    There are many conditions we could be running under:
    1) User wants to run locally on their own machine.
    2) User wants to run on the cluster.
        a) User starts job from their machine ("remote launch").
            Consequently, this function gets called on the
                i) user machine (to copy resume-state and code to the head node),
                ii) head node (to prepare the job and launch the tasks) and
                iii) cluster nodes (to execute the tasks).
        b) User starts job from the cluster head node.
            This function gets called on ii and iii above.

    :param JobInfo:
    :param ji_kwargs:
    :param resume:
    :param result_dir:
    :param delete_tmp:
    :param max_tasks:
    :param max_task_time:
    :param mem_per_task:
    :param poll_interval:
    :param username:
    :return:
    """
    on_cluster = snep.configuration.on_cluster
    on_head = snep.configuration.on_head

    print('on_head: {} on_cluster: {}'.format(on_head, on_cluster))

    config_cluster = snep.configuration.config['cluster']
    script_name = os.path.basename(sys.argv[0])
    job_name = script_name.replace('.py', '')
    job_dir = snep.configuration.config['jobdir']

    if config_cluster and not (on_cluster or on_head):
        from paramiko import SSHClient, AutoAddPolicy
        print('RUNNING REMOTE')
        # Run remote
        client = SSHClient()
        client.set_missing_host_key_policy(AutoAddPolicy())
        client.load_host_keys(os.path.expanduser('~/.ssh/known_hosts'))
        client.connect(snep.configuration.cluster_url, username=username)

        def run_remote(c, print_cmd=True, print_out=True, interactive=False):
            if print_cmd:
                print('({}-to-head): '.format(snep.configuration.hostname), c)
            if interactive:
                chan = client.get_transport().open_session(timeout=None)
                chan.get_pty(width=240)
                chan.settimeout(None)
                chan.exec_command(c)
                bufsize = -1
                stdin = chan.makefile('wb', bufsize)
                stdout = chan.makefile('r', bufsize)
                stderr = chan.makefile_stderr('r', bufsize)
            else:
                stdin, stdout, stderr = client.exec_command(c, get_pty=interactive)
            if print_out:
                for l in stdout:
                    print('(head-stdout): '+l, end='')
            return stdout
        run_cmd = run_remote
    else:
        print('RUNNING LOCAL')
        run_cmd = run_local
        run_remote = run_local

    remote_launch = '' != job_dir

    print('HOSTNAME ----- '+snep.configuration.hostname)
    if remote_launch:
        print('JOBDIR ----- '+job_dir)

    if not remote_launch:
        ts = time.strftime('%Y-%m-%d-%Hh%Mm%Ss')
        job_dir = '~/{result_dir}/{job_name}/{ts}'.format(result_dir=result_dir, job_name=job_name, ts=ts)
        run_cmd('mkdir -p {job_dir}'.format(job_dir=job_dir))
    ji_kwargs['job_dir'] = job_dir
    temp_dir = "{job_dir}/tmp".format(job_dir=job_dir)
    state_path = None
    # If we are currently on a user machine,
    # or starting a job from the head node then we need to handle temp file stuff.
    manage_temp = not (on_cluster or remote_launch)
    if manage_temp:
        run_cmd('mkdir {temp_dir}'.format(temp_dir=temp_dir))
        if resume:
            state_path = JobInfo.extract_state(job_name, result_dir)
            if config_cluster:
                # make temp-dir on head, copy code and previous state to it
                cmd = "scp -r {state_path} {url}:{temp_dir}".format(state_path=state_path,
                                                                    url=snep.configuration.cluster_url,
                                                                    temp_dir=temp_dir)
            else:
                cmd = 'cp {state_path} {temp_dir}'.format(state_path=state_path, temp_dir=temp_dir)
            run_local(cmd)
        # copy code
        if config_cluster:
            # cmd = "scp -r * {url}:{temp_dir}".format(url=snep.configuration.cluster_url, temp_dir=temp_dir)
            src = "./*"
            dst = "{url}:{temp_dir}".format(url=snep.configuration.cluster_url, temp_dir=temp_dir)
            # use rsync, options: recursive, verbose, compressed
            cmd = "rsync -rzv --exclude='.git*' --exclude='*.pyc' --exclude='__pycache__' {} {}".format(src, dst)
            run_local(cmd, print_out=False)

    user_cmd = '' if conda_env is None else 'source activate {conda_env} && '.format(conda_env=conda_env)

    if config_cluster and not (on_head or on_cluster):
        # run code remotely
        py_cmd = 'python {script_name} --jobdir {job_dir}'.format(script_name=script_name, job_dir=job_dir)
        cmd = "screen /bin/bash -c 'cd {temp_dir} && {user_cmd} {py_cmd}'".format(temp_dir=temp_dir, user_cmd=user_cmd,
                                                                                  py_cmd=py_cmd)
        # print(cmd)
        run_remote(cmd, interactive=True, print_out=True)
    elif not config_cluster or on_head:
        # Generate param space, write params/state and run (local or array task).
        # wait for task to finish, collect results and write to out-dir
        # NEEDS: temp-dir, out-dir
        from .configuration import config
        job_info = JobInfo(**ji_kwargs)
        check_path = os.path.join(temp_dir, 'experiment.h5')
        if not state_path and os.path.exists(check_path):
            state_path = check_path
        # state_path = os.environ.get('SNEPSTATE', state_path)
        job_info.resume_state(state_path)
        _backup_code(job_info.job_dir, additional_files)

        sys_params = _get_sys_params(temp_dir)
        task_ids = job_info.prepare_tasks()
        n_tasks = len(task_ids)

        all_sys_params = {job_info.task_name(task_id): _add_sys_params_for_task(job_info, sys_params, task_id, tid)
                          for tid, task_id in enumerate(task_ids, 1)}

        for tid, task_id in enumerate(task_ids, 1):
            task_info = job_info.get_task_info(task_id)
            task_name = job_info.task_name(task_id)
            task_sys_params = all_sys_params[task_name]
            kwargs = dict(task_id=task_id, run_task=JobInfo.run_task, task_info=task_info, sys_params=task_sys_params)
            with open(os.path.expanduser(task_sys_params['params_file_path']), 'wb') as f:
                # cPickle.dump(kwargs, f, cPickle.HIGHEST_PROTOCOL)
                p = pickle.Pickler(f, protocol=pickle.HIGHEST_PROTOCOL)
                p.dump(kwargs)
        if on_head:
            # submit array task
            print('Running on head, submit array task')
            cmd = 'qsub -t 1-{n_tasks} -V -q cognition-all.q -cwd'.format(n_tasks=n_tasks)
            cmd += ' -l h_vmem={vmem}G'.format(vmem=mem_per_task)
            if max_tasks is not None:
                cmd += ' -tc {max_tasks}'.format(max_tasks=max_tasks)
            if max_task_time is not None:
                cmd += ' -l h_rt={job_time}'.format(job_time=max_task_time)
            if config['exclude_nodes']:
                cmd += ' -l h="' + '&'.join(('!cognition{:02}*'.format(x) for x in config['exclude_nodes'])) + '"'
            cmd += ' -N {job_name}  -b y'.format(job_name=job_name)
            cmd += ' -v SNEPJOBDIR={job_dir}'.format(job_dir=job_dir)
            if resume:
                cmd += ' SNEPSTATE={temp_dir}'.format(temp_dir=temp_dir)
            # cmd += ' -o {jobdir}/stdout -e {jobdir}/stderr'.format(jobdir=jobdir)
            # cmd += 'source activate idp; '
            cmd += ' "{user_cmd} python {script_name}"'.format(user_cmd=user_cmd, script_name=script_name)
            # submit and wait for the result
            job_num = _do_qsub(cmd, poll_interval, task_ids, job_info, job_name, all_sys_params, delete_tmp)
            # all_task_info = _get_task_info_for_job(job_num)
        else:
            # run local
            print('Running locally.')
            start = time.time()
            tdr = os.path.expanduser(temp_dir)
            all_tasks = []
            for task_params in all_sys_params.values():
                kwargs = dict()
                with open(task_params['params_file_path'], 'rb') as f:
                    # p = pickle.Pickler(f, protocol=pickle.HIGHEST_PROTOCOL)
                    p = pickle.Unpickler(f)
                    kwargs.update(p.load())
                all_tasks.append((_run_single_task, kwargs, max_task_time))

            num_proc = config.prompt_procs()
            # case without multi process
            # for n, (rst, kwargs, max_task_time) in enumerate(all_tasks):
            #     task_id, run_time, peak_memory = rst(**kwargs)
            pool = mp.Pool(processes=num_proc, maxtasksperchild=1)
            for n, (task_id, run_time, peak_memory) in enumerate(pool.imap_unordered(_run_with_timeout, all_tasks), 1):
                task_name = job_info.task_name(task_id)
                print(160*'=', '\nFinished task {} of {} {}'.format(n, len(task_ids), task_name))
                print('Used {peak_memory} of {mem_per_task} GB ({frac})'.format(peak_memory=peak_memory,
                                                                                mem_per_task=mem_per_task,
                                                                                frac=peak_memory/mem_per_task))
                print(160*'=')
            finish = time.time()
            print('Total job ran in {:.1f}'.format((finish - start)/60., 'minutes'))
            os.chdir(tdr)

            # Collect results from temp dirs.
            for tid, task_id in enumerate(task_ids, 1):
                task_name = job_info.task_name(task_id)
                _collect_result(job_info, None, job_name, task_name,
                                all_sys_params[task_name]['taskdir'], all_sys_params[task_name]['tempdir_task'],
                                tid, task_id, all_sys_params[task_name]['result_file_path'], delete_tmp, False)
    elif on_cluster:
        # Run task
        # NEEDS: tmp-dir
        tid = os.environ['SGE_TASK_ID']
        print('Running task {tid}'.format(tid=tid))
        timeout = None
        # print('Running task on cluster {tid}.'.format(tid=tid))
        task_params_path = './{tid}/params.pickle'.format(tid=tid)
        with open(task_params_path, 'rb') as f:
            p = pickle.Unpickler(f)
            kwargs = p.load()
            print('Results saved in {}'.format(kwargs['sys_params']['result_file_path']))
            task_id, run_time, peak_memory = _run_with_timeout((_run_single_task, kwargs, timeout))

    if manage_temp and delete_tmp:
        # delete main temp-dir
        try:
            # time.sleep(10.)
            run_cmd('rm -rf {temp_dir}'.format(temp_dir=temp_dir))
        except Exception as e:
            print('Failed to delete temp dir: {}'.format(e))

    if config_cluster and not (on_head or on_cluster):
        import six.moves
        # copy results back
        run_local('scp -r {url}:{job_dir} ~/{result_dir}/{job_name}'.format(
            url=snep.configuration.cluster_url, job_dir=job_dir, result_dir=result_dir, job_name=job_name))

        if six.moves.input('Delete results from cluster? y/[n] ') == 'y':
            run_remote('rm -rf {job_dir}'.format(job_dir=job_dir))

    if state_path and not (on_head or on_cluster):
        sfn = os.path.join('~', result_dir, job_name, os.path.basename(job_dir), os.path.basename(os.path.dirname(state_path)))
        run_local('ln -s {tfn} {sfn}'.format(tfn=os.path.dirname(state_path), sfn=sfn))


def _collect_result(job_info, job_num, job_name, task_name, taskdir, tempdir, tid, task_id, result_file_path,
                    delete_tmp, on_cluster, sge_task_info=None):
    import shutil
    from snep.tables.data import open_data_file
    try:
        # with open(result_file_path, 'rb') as f:
        #     p = pickle.Unpickler(f)
        #     results = p.load()
        with open_data_file(result_file_path) as f:
            results = SimulationResult.from_dict(f.read_data_root())

    except (IOError, EOFError, KeyError) as e:
        err_str = 'Failed to load results file {tid} {tn}: '.format(tid=tid, tn=task_name)
        err_str += str(e)
        finaldata = {'log_file':{'exc_info': err_str

                          }
                     }
        results = SimulationResult(task_id, finaldata, 'error', 0)
    if job_num:
        for eo, name in [('e', 'stderr'), ('o', 'stdout')]:
            str_stuff = dict(job_name=job_name, job_num=job_num, tid=tid, eo=eo)
            std_fn = '{job_name}.{eo}{job_num}.{tid}'.format(**str_stuff)
            try:
                with open(std_fn) as f:
                    std = f.read()
                    results.finaldata.setdefault('log_file', {})[name] = std
            except IOError:
                print('Output file not found: {}'.format(std_fn))
        if on_cluster:
            if int(tid) in sge_task_info:
                results.cluster_info = sge_task_info[int(tid)]
            else:
                print('No SGE task info found for task', tid)
    # print(results.status, results.tasktime, results.cluster_info)
    job_info.record_result(task_id, results, taskdir)

    try:  # Delete the tempdir created in _add_sys_params_for_task
        if delete_tmp:
            time.sleep(2.)
            shutil.rmtree(tempdir)
    except Exception as e:
        print('Error deleting temp dir {}, {}'.format(tempdir, e))


class SimulationResult(object):
    def __init__(self, task_id, finaldata, status, tasktime):
        self.task_id = task_id
        self.finaldata = finaldata
        self.status = status
        self.tasktime = tasktime
        self.cluster_info = None

    def to_dict(self):
        result = {'finaldata': self.finaldata,
                  'status': self.status,
                  'tasktime': np.array([self.tasktime]),
                  }
        if self.cluster_info:
            result['cluster_info'] = self.cluster_info
        return result

    @staticmethod
    def from_dict(d):
        final = d['finaldata'] if 'finaldata' in d else {'raw_data': {'failed': np.zeros(1)},
                                                         'computed': {'failed': np.zeros(1)},
                                                         'sim_state': {'failed': np.zeros(1)}}
        status = d['status'] if 'status' in d else 'status?'
        tasktime = d['tasktime'] if 'tasktime' in d else np.array([-1])
        sr = SimulationResult(0, final, status, tasktime)
        if 'cluster_info' in d:
            sr.cluster_info = d['cluster_info']
        return sr


def _run_with_timeout(args):
    import traceback
    import threading
    import resource
    from snep.tables.data import open_data_file
    target, kwargs, timeout = args
    task_id = kwargs['task_id']
    # result_q = kwargs['result_q']
    time_start = time.time()
    result = None
    try:
        if timeout:
            t = threading.Thread(target=target, kwargs=kwargs)
            t.start()
            tid = t.ident
            t.join(timeout)
            timedout = t.is_alive()
            if timedout:
                all_frames = sys._current_frames()
                if tid in all_frames:
                    stack = all_frames[tid]
                    # filename, linenum, funcname, code
                    msg = 'File: "{0}", line {1}, in {2}, code: {3}'
                    callstack = [msg.format(fn,ln,f,c.strip() if c else 'None')
                                    for fn,ln,f,c in traceback.extract_stack(stack)]
                    finaldata = {'log_file': {'callstack':callstack}}
                else:
                    finaldata = {'log_file': {'callstack':'Thread info not available.'}}
                result = SimulationResult(task_id, finaldata, 'timedout', time.time() - time_start)
                # result_q.put(result, block=True)
        else:
            result = target(**kwargs)
    except:
        traceback.print_exc()
        finaldata = {'log_file': {'exc_info': traceback.format_exc()}}
        result = SimulationResult(task_id, finaldata, 'error', time.time() - time_start)
        # result_q.put(result, block=True)

    if result is not None:
        result_file_path = kwargs['sys_params']['result_file_path']
        print('Target function returned a value, or threw an exception! Writing to', result_file_path)
        print(str(result))

        with open_data_file(result_file_path) as f:
            f.store_data_root(result.to_dict())
        # with open(result_file_path, 'wb') as f:
        #     # cPickle.dump(result, f, cPickle.HIGHEST_PROTOCOL)
        #     p = pickle.Pickler(f, protocol=pickle.HIGHEST_PROTOCOL)
        #     p.dump(result)

    peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 10**6
    run_time = time.time()-time_start
    return task_id, run_time, peak_memory

def _run_single_task(task_id, run_task, task_info, sys_params):
    """
    The function used to run a simulation in a subprocess.
    """
    import numpy, random
    import traceback, tempfile
    if 'seed' in task_info:
        numpy.random.seed(int(task_info['seed']))
        random.seed(int(task_info['seed']))

    tempdir_original = sys_params['tempdir_original']
    tempdir_task = sys_params['tempdir_task']
    PYTHONCOMPILED_original = sys_params['PYTHONCOMPILED_original']
    PYTHONCOMPILED_subproc  = sys_params['PYTHONCOMPILED_subproc']

    cwd = os.getcwd()

    exception_info = ''
    result = {}
    try:
        # Change the temp directories used to build the weave stuff.
        # Without this the build will fail, due to weave_imp.o being accessed
        # by multiple processes.
        # if os.path.exists(tempdir_task):
        #     shutil.rmtree(tempdir_task)
        # os.mkdir(tempdir_task)
        assert os.path.exists(tempdir_task), "The temp dir {} does not exist for task {}".format(tempdir_task, task_id)
        tempfile.tempdir = tempdir_task
        os.environ['PYTHONCOMPILED'] = PYTHONCOMPILED_subproc
    except:
        traceback.print_exc()
        exception_info += '\nEXCEPTION SETTING TEMPDIRS: {0}'.format(traceback.format_exc())

    t_start = time.time()
    try:
        result = run_task(task_info, sys_params['taskdir'], sys_params['tempdir_task'])
    except:
        traceback.print_exc()
        exception_info += '\nEXCEPTION IN RUN: {0}'.format(traceback.format_exc())
        status = 'error'
    else:
        status = 'finished'
    t_end = time.time()
    tasktime = t_end - t_start

    os.chdir(cwd)  # Make sure we restore the original CWD

    try:
        if PYTHONCOMPILED_original:
            os.environ['PYTHONCOMPILED'] = PYTHONCOMPILED_original
        elif 'PYTHONCOMPILED' in os.environ:
            del os.environ['PYTHONCOMPILED']
    except:
        traceback.print_exc()
        exception_info += '\nEXCEPTION RESETTING PYTHONCOMPILED: {0}'.format(traceback.format_exc())

    try:
        tempfile.tempdir = tempdir_original
        # Do not delete tmp directory, it will be deleted after entire job is done
        # shutil.rmtree(tempdir_task)
    except:
        traceback.print_exc()
        exception_info += '\nEXCEPTION RESETTING TEMPDIR: {0}'.format(traceback.format_exc())

    if exception_info != '':
        result.setdefault('log_file', {'exc_info': ''})['exc_info'] += exception_info
    result = SimulationResult(task_id, result, status, tasktime)
    # try:
    #     result_q.put(simresult, block=True)
    # except IOError:
    #     traceback.print_exc()
    #     simresult.status = 'error_ipc'
    #     log_file = simresult.finaldata.setdefault('log_file', {})
    #     exc_info = log_file.setdefault('exc_info', '')
    #     exc_info += traceback.format_exc()
    #     result_q.put(simresult, block=True)
    result_file_path = sys_params['result_file_path']

    with open_data_file(result_file_path) as f:
        f.store_data_root(result.to_dict())

    # with open(result_file_path, 'wb') as f:
    #     # cPickle.dump(result, f, cPickle.HIGHEST_PROTOCOL)
    #     p = pickle.Pickler(f, protocol=pickle.HIGHEST_PROTOCOL)
    #     p.dump(result)
    return  # Don't return anything


def _get_sys_params(temp_dir):
    import tempfile

    tempdir_orig = tempfile.gettempdir()
    # user_time = str(os.getuid()) + '{0:.6f}'.format(time.time())
    # tempdir_stem = os.path.join(tempdir_orig, user_time)

    sys_params = {'tempdir_original': tempdir_orig, 'tempdir_stem': os.path.expanduser(temp_dir)}

    if 'PYTHONCOMPILED' in os.environ:
        PYTHONCOMPILED_original = os.environ['PYTHONCOMPILED']
    else:
        PYTHONCOMPILED_original = None
    sys_params['PYTHONCOMPILED_original'] = PYTHONCOMPILED_original
    return sys_params


def _add_sys_params_for_task(job_info, sys_params, task_id, tid):
    """
    Add task specific
    """
    task_sys_params = dict(sys_params)

    tn = job_info.task_name(task_id)
    task_sys_params['task_name'] = tn # used to be results_group
    task_sys_params['taskdir'] = os.path.join(job_info.job_dir, tn) # used to be subprocdir

    tempdir_stem = task_sys_params['tempdir_stem']
    tempdir_task = os.path.join(tempdir_stem, str(tid))
    task_sys_params['tempdir_task'] = tempdir_task
    os.mkdir(tempdir_task)  # Temp dir deleted in _collect_result
    task_sys_params['result_file_path'] = os.path.join(tempdir_task, 'results.h5')
    task_sys_params['params_file_path'] = os.path.join(tempdir_task, 'params.pickle')

    PYTHONCOMPILED_original = task_sys_params['PYTHONCOMPILED_original']
    PYTHONCOMPILED_subproc = (tempdir_task,PYTHONCOMPILED_original) if PYTHONCOMPILED_original else (tempdir_task,)
    task_sys_params['PYTHONCOMPILED_subproc'] = (';' if (sys.platform=='win32') else ':').join(PYTHONCOMPILED_subproc)

    return task_sys_params


def _backup_code(job_dir, additional_files):
    import shutil
    code_dir = os.path.expanduser(os.path.join(job_dir, 'code'))
    assert not os.path.exists(code_dir)
    os.mkdir(code_dir)

    script_file = os.path.join(os.getcwd(), sys.argv[0])
    if not additional_files:
        additional_files = []
    additional_files.append(script_file)

    for af in additional_files:
        shutil.copy2(af, code_dir)

    snep_dir = os.path.dirname(os.path.realpath(__file__))
    ignore = shutil.ignore_patterns('*.pyc', '.git', '.svn', '.idea')
    shutil.copytree(snep_dir, os.path.join(code_dir, 'snep'), ignore=ignore)


def _do_qsub(cmd, poll_interval, task_ids, job_info, job_name, all_sys_params, delete_tmp):
    import re
    n_tasks = len(task_ids)
    start = time.time()
    stdout = run_local(cmd, print_out=True)
    jids = re.findall('[0-9]{5,7}', str(stdout))
    if len(jids) < 1:
        raise Exception("qsub resulted in: {}".format(stdout))

    time.sleep(1)

    stdout = run_local('qstat', print_out=True)
    job_num = jids[0]
    print('Started job {}'.format(job_num))
    try:
        uncollected = set(str(i) for i in range(1, n_tasks+1))
        was_running = set()
        sge_task_info = None

        task_re = re.compile(job_num + ' .+ r .+ ([0-9]{1,10})$', re.MULTILINE)

        while job_num in str(stdout):
            time.sleep(poll_interval*60)
            stdout = run_local('qstat', print_out=True)

            is_running = set(re.findall(task_re, str(stdout)))
            finished = was_running.difference(is_running)
            if finished:
                sge_task_info = _get_task_info_for_job(job_num)
            for tid in finished:
                task_id = task_ids[int(tid) - 1]
                task_name = job_info.task_name(task_id)
                taskdir = all_sys_params[task_name]['taskdir']
                tempdir = all_sys_params[task_name]['tempdir_task']
                result_file_path = all_sys_params[task_name]['result_file_path']
                print('Collecting result during run for {}'.format(task_name))
                _collect_result(job_info, job_num, job_name, task_name, taskdir, tempdir,
                                tid, task_id, result_file_path, delete_tmp, True, sge_task_info)
                uncollected.remove(tid)

            print('Found {} running tasks. {} tasks finished just now. {} tasks queued or uncollected.'.format(
                len(is_running), len(finished), len(uncollected)))

            was_running = is_running


        if uncollected:
            sge_task_info = _get_task_info_for_job(job_num)
        for tid in uncollected:
            task_id = task_ids[int(tid) - 1]
            task_name = job_info.task_name(task_id)
            taskdir = all_sys_params[task_name]['taskdir']
            tempdir = all_sys_params[task_name]['tempdir_task']
            result_file_path = all_sys_params[task_name]['result_file_path']
            print('Collecting result after run for {}'.format(task_name))
            _collect_result(job_info, job_num, job_name, task_name, taskdir, tempdir,
                            tid, task_id, result_file_path, delete_tmp, True, sge_task_info)

    except KeyboardInterrupt:
        run_local('qdel {}'.format(job_num))
        m = 'aborted'
    else:
        m = 'completed'
    t = time.time() - start
    print('Job {2} at {0} after {1:.1f} minutes!'.format(time.strftime('%m-%d-%Hh%Mm%Ss'), t/60., m))
    return job_num


def _get_task_info_for_job(job_num, tid=None):
    import re
    t_info = [('taskid',      '(\d+)',             int),
              ('qsub_time',   '(.+)',              str),
              ('start_time',  '(.+)',              str),
              ('end_time',    '(.+)',              str),
              ('failed',      '([01])',            int),
              ('exit_status', '(\d+)',             int),
              ('maxvmem',     '(\d*\.\d*)([MG]?)', float)]
    all_task_info, curr, t_i = dict(), dict(), list(t_info)
    user_name = run_local('id -u -n', print_out=False)
    if tid:
        cmd = 'qacct -j {job_num} -t {tid} -o {user_name}'.format(job_num=job_num, user_name=user_name, tid=tid)
    else:
        cmd = 'qacct -j {job_num} -o {user_name}'.format(job_num=job_num, user_name=user_name)

    # print('Running qacct can take a while')
    r = run_local(cmd, print_out=False)
    for l in r.splitlines():
        if re.match(60*'=', l):
            curr, t_i = dict(), list(t_info)
        else:
            for k, p, t in t_i:
                g = re.match(k+'\s+'+p, l)
                if g:
                    v = t(g.group(1))
                    if k == 'taskid':
                        all_task_info[v] = curr
                    else:
                        curr[k] = v/1e3 if (k == 'maxvmem' and g.group(2) == 'M') else v
                    t_i.remove((k, p, t))
                    break
    t_i = set([v[0] for v in t_info if v[0] != 'taskid'])
    for k, v in all_task_info.items():
        x = t_i.difference(v.keys())
        if len(x):
            print('Task {} missing: {}'.format(k, list(x)))
    return all_task_info
