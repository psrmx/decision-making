# from __future__ import print_function
# import os, gc, time, threading, sys
# from fabric.api import local as fab_local, run as fab_run, cd as fab_cd, settings as fab_settings
# import snep.configuration
#
#
# def run(JobInfo, ji_kwargs, resume=False, job_time=None, mem_per_task=None, timeout=None,
#         remote_codedir='snep_projects', resultdir='experiments', qsub_env=''):
#     """
#     JobInfo - a class that constructs objects using ji_kwargs, and implements the following methods/attributes:
#         job_dir - an attribute that can be set to the current job directory
#         prepare_tasks(JobInfo) - prepare task info, store it in JobInfo and return a list of task_ids
#         get_task_info(JobInfo, task_id) - returns task_info from JobInfo for task_id
#         task_name(task_id) - returns a unique string based upon the task_id
#         record_result(JobInfo, simresult) - records the result of the task contained in a SimulationResult object
#         log_info/err - functions that take a msg to be logged
#         run_task(task_info) - classmethod that does the actual work of running a task described in task_info
#     """
#
#     ji_kwargs['job_dir'] = snep.configuration.config['jobdir']
#     config_mpi = snep.configuration.config['cluster']
#     on_cluster = snep.configuration.on_cluster
#     on_head = snep.configuration.on_head
#     job_name = os.path.basename(sys.argv[0]).replace('.py', '')
#
#     if resume and not on_cluster:
#         statepath = JobInfo.extract_state(job_name, resultdir)
#         if config_mpi and not on_head:
#             statepath = _copy_state_to_cluster(job_name, statepath)
#     else:
#         statepath = None
#
#     job_info = None
#     if config_mpi and not (on_cluster or on_head):
#         _run_remote(job_name, remote_codedir, resultdir, qsub_env, mem_per_task, job_time, statepath)
#     elif config_mpi:
#         job_info = _run_mpi(JobInfo, ji_kwargs, timeout)
#     else:
#         job_info = _run_local(JobInfo, ji_kwargs, timeout, statepath)
#     if job_info:
#         delete_prev = 'STATE' in os.environ
#         job_info.finalize(delete_prev)
#     return job_info
#
#
# def _copy_state_to_cluster(job_name, statepath):
#     import tempfile
#     with tempfile.NamedTemporaryFile(prefix = job_name) as tf:
#         remotedir = "~/{remotedir}".format(remotedir=os.path.basename(tf.name))
#     with fab_settings(host_string=snep.configuration.cluster_url):
#         fab_run('mkdir {remotedir}'.format(remotedir=remotedir))
#     fab_local("scp -r {statepath} ge:{remotedir}".format(statepath=statepath, remotedir=remotedir))
#     return remotedir
#
#
# class SimulationResult(object):
#     def __init__(self, task_id, finaldata, status, tasktime):
#         self.task_id = task_id
#         self.finaldata = finaldata
#         self.status = status
#         self.tasktime = tasktime
#
#
# def enum(*sequential, **named):
#     """Handy way to fake an enumerated type in Python
#     http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
#     """
#     enums = dict(zip(sequential, range(len(sequential))), **named)
#     return type('Enum', (), enums)
#
# # Define MPI message tags
# tags = enum('READY', 'DONE', 'EXIT', 'START', 'NONE')
# start_worker = '--worker'
# """
# https://github.com/jbornschein/mpi4py-examples/blob/master/09-task-pull.py
# https://github.com/jbornschein/mpi4py-examples/blob/master/10-task-pull-spawn.py
# """
#
#
# def _run_mpi(JobInfo, ji_kwargs, timeout):
#     if snep.configuration.config['worker']:
#         _mpi_worker(timeout)
#         return None
#     else:
#         return _mpi_parent(JobInfo, ji_kwargs)
#
#
# def _mpi_worker(timeout):
#     from mpi4py import MPI
#     import multiprocessing as mp
#
#     manager    = mp.Manager()
#     result_q   = manager.Queue()
#
#     Status = getattr(MPI, 'Status')
#     status = Status()
#     MPI_ANY_TAG = getattr(MPI, 'ANY_TAG')
#     # Connect to parent
#     try:
#         MPI_Comm = getattr(MPI, 'Comm')
#         comm = MPI_Comm.Get_parent()
#         _rank = comm.Get_rank()
#     except:
#         raise ValueError('Could not connect to parent: Program should be started without argument')
#
#     tag = tags.NONE
#     while tag != tags.EXIT:
#         comm.send(None, dest=0, tag=tags.READY)
#         kwargs = comm.recv(source=0, tag=MPI_ANY_TAG, status=status)
#         tag = status.Get_tag()
#
#         if tag == tags.START:
#             assert kwargs['result_q'] is None
#             kwargs['result_q'] = result_q
#             _run_with_timeout((_run_single_task, kwargs, timeout))
#             result = result_q.get(block=True)
#             comm.send(result, dest=0, tag=tags.DONE)
#
#     comm.send(None, dest=0, tag=tags.EXIT)
#
#     # Shutdown
#     comm.Disconnect()
#
#
# def _mpi_parent(JobInfo, ji_kwargs):
#     from snep.configuration import config
#     from mpi4py import MPI
#
#     job_info = JobInfo(**ji_kwargs)
#     job_info.resume_state(os.environ.get('STATE', None))
#     _backup_code(job_info.job_dir)
#     num_workers = config['procs']# - 1
#
#     Status = getattr(MPI, 'Status')
#     status = Status()
#     MPI_ANY_TAG = getattr(MPI, 'ANY_TAG')
#     COMM_WORLD = getattr(MPI, 'COMM_WORLD')
#     Wtime = getattr(MPI, 'Wtime')
#     MPI_ANY_SOURCE = getattr(MPI, 'ANY_SOURCE')
#
#     sys_params = _get_sys_params()
#
#     # Start clock
#     start = Wtime()
#     task_ids = job_info.prepare_tasks()
#     n_tasks = len(task_ids)
#
#     all_sys_params = {job_info.task_name(task_id): _add_sys_params_for_task(job_info, sys_params, task_id)
#                       for task_id in task_ids}
#     # Spawn workers
#     comm = COMM_WORLD.Spawn(
#         sys.executable,
#         args=[sys.argv[0], start_worker],
#         maxprocs=num_workers)
#
#     # Reply to whoever asks until done
#     closed_workers = 0
#     task_index = 0
#     job_info.log_info("Master starting with {} workers (one process is needed for the parent)".format(num_workers))
#     while closed_workers < num_workers:
#         data = comm.recv(source=MPI_ANY_SOURCE, tag=MPI_ANY_TAG, status=status)
#         source = status.Get_source()
#         tag = status.Get_tag()
#         if tag == tags.READY:
#             # Worker is ready, so send it a task
#             if task_index < n_tasks:
#                 task_id = task_ids[task_index]
#                 task_name = job_info.task_name(task_id)
#                 task_info = job_info.get_task_info(task_id)
#                 sys_params = all_sys_params[task_name]
#                 kwargs = dict(task_id=task_id, run_task=JobInfo.run_task, task_info=task_info, result_q=None, sys_params=sys_params)
#
#                 comm.send(kwargs, dest=source, tag=tags.START)
#                 job_info.log_info("Sending task {} to worker {}".format(task_name, source))
#                 task_index += 1
#             else:
#                 comm.send(None, dest=source, tag=tags.EXIT)
#         elif tag == tags.DONE:
#             results = data
#             job_info.record_result(results, all_sys_params[job_info.task_name(results.task_id)]['taskdir'])
#             job_info.log_info("Got data from worker {}".format(source))
#         elif tag == tags.EXIT:
#             job_info.log_info("Worker {} exited.".format(source))
#             closed_workers += 1
#
#
#     # Final statistics
#     job_info.log_info('Processed in {:.2f} secs'.format(Wtime() - start))
#
#     # Shutdown
#     comm.Disconnect()
#
#     return job_info
#
#
# def _run_local(JobInfo, ji_kwargs, timeout, statepath):
#     from snep.configuration import config
#     import multiprocessing as mp
#     manager = mp.Manager()
#     result_q = manager.Queue()
#
#     '''
#     This is responsible for actually distributing the simulations to subprocesses for execution.
#     '''
#     job_info = JobInfo(**ji_kwargs)
#     job_info.resume_state(statepath)
#     task_ids = job_info.prepare_tasks()
#     num_sims = len(task_ids)
#     _backup_code(job_info.job_dir)
#
#     sys_params = _get_sys_params()
#
#     job_info.log_info('Starting in master pid: {0}, thr: {1}'.format(os.getpid(),
#                                                                      threading.current_thread().name))
#
#     all_sys_params = {job_info.task_name(task_id): _add_sys_params_for_task(job_info, sys_params, task_id)
#                       for task_id in task_ids}
#     all_tasks = ((_run_single_task,
#                   dict(task_id=task_id,
#                        run_task=JobInfo.run_task,
#                        task_info=job_info.get_task_info(task_id),
#                        result_q=result_q,
#                        sys_params=all_sys_params[job_info.task_name(task_id)]),
#                   timeout)
#                  for task_id in task_ids)
#
#     pool = None
#     num_procs = config['procs']
#     if config['debug']:
#         start_log = "Starting {0} simulations sequentially without multiprocessor\n".format(num_sims)
#         results = (_run_with_timeout(ns) for ns in all_tasks)
#     else:
#         num_procs = 1
#         pool = mp.Pool(num_procs,maxtasksperchild=1)
#         results = pool.imap_unordered(_run_with_timeout, all_tasks)
#         start_log = "Starting multiprocessing pool with {0} processes for {1} simulations.\n".format(num_procs,num_sims)
#
#     # measure time of all simulations
#     global_start=time.time()
#
#     job_info.log_info(start_log)
#     timedout_ids = []
#     error_ids = []
#     single_times = []
#     log = {'ns':num_sims,
#            'div':'================================================================'}
#     for log['fc'], (task_id, single_time) in enumerate(results,1):
#         elapsed_time = (time.time() - global_start) / 60.
#         result = result_q.get(block=True)
#         job_info.record_result(result, all_sys_params[job_info.task_name(task_id)]['taskdir'])
#         if result.status == 'timedout':
#             timedout_ids.append(task_id)
#             msg = '{elh}h{elm}m minutes, {fc}/{ns}. !!! SIMULATION TIMED OUT !!! '
#         elif result.status == 'error':
#             error_ids.append(task_id)
#             msg = '{elh}h{elm}m minutes, {fc}/{ns}. !!! SIMULATION THREW EXCEPTION !!! '
#         else:
#             msg = '{elh}h{elm}m minutes elapsed, {fc}/{ns} simulations complete. '
#         est = '~{esh}h{esm}m remain\n{div}'
#
#         single_times.append(single_time / 60.)
#         estimated_time = _estimate_time_remaining(single_times, elapsed_time,
#                                              num_sims, num_procs)
#         log['esh'],log['esm'] = int(estimated_time/60),int(estimated_time%60)
#         log['elh'],log['elm'] = int(elapsed_time  /60),int(elapsed_time  %60)
#         job_info.log_info(msg.format(**log) + est.format(**log))
#         gc.collect()
#
#     if pool:
#         pool.close()
#         del pool
#
#     for task_id in timedout_ids:
#         pt_name = job_info.task_name(task_id)
#         job_info.log_err('TIMED OUT: {0}'.format(pt_name))
#
#     for task_id in error_ids:
#         pt_name = job_info.task_name(task_id)
#         job_info.log_err('EXCEPTION: {0}'.format(pt_name))
#
#     global_end=time.time()
#     job_info.log_info("...{0} simulations done in {1:.1f} minutes.".format(num_sims,
#                                                            (global_end-global_start)/60.))
#
#     job_info.log_info("Results stored in {0}".format(job_info.job_dir))
#
#     return job_info
#
#
# def _run_with_timeout((target, kwargs, timeout)):
#     import traceback
#     task_id = kwargs['task_id']
#     result_q = kwargs['result_q']
#     time_start = time.time()
#     try:
#         if timeout:
#             t = threading.Thread(target=target, kwargs=kwargs)
#             t.start()
#             tid = t.ident
#             t.join(timeout)
#             timedout = t.is_alive()
#             if timedout:
#                 all_frames = sys._current_frames()
#                 if tid in all_frames:
#                     stack = all_frames[tid]
#                     # filename, linenum, funcname, code
#                     msg = 'File: "{0}", line {1}, in {2}, code: {3}'
#                     callstack = [msg.format(fn,ln,f,c.strip() if c else 'None')
#                                     for fn,ln,f,c in traceback.extract_stack(stack)]
#                     finaldata = {'log_file': {'callstack':callstack}}
#                 else:
#                     finaldata = {'log_file': {'callstack':'Thread info not available.'}}
#                 result = SimulationResult(task_id, finaldata, 'timedout', time.time() - time_start)
#                 result_q.put(result, block=True)
#         else:
#             target(**kwargs)
#     except:
#         traceback.print_exc()
#         finaldata = {'log_file': {'exc_info': traceback.format_exc()}}
#         result = SimulationResult(task_id, finaldata, 'error', time.time() - time_start)
#         result_q.put(result, block=True)
#
#     run_time = time.time()-time_start
#     return task_id, run_time
#
#
# def _run_single_task(task_id, run_task, task_info, result_q, sys_params):
#     """
#     The function used to run a simulation in a subprocess.
#     """
#     import numpy, random, shutil
#     import traceback, tempfile
#     if 'seed' in task_info:
#         numpy.random.seed(int(task_info['seed']))
#         random.seed(int(task_info['seed']))
#
#     tempdir_original = sys_params['tempdir_original']
#     tempdir_subproc = sys_params['tempdir_subproc']
#     PYTHONCOMPILED_original = sys_params['PYTHONCOMPILED_original']
#     PYTHONCOMPILED_subproc  = sys_params['PYTHONCOMPILED_subproc']
#
#     cwd = os.getcwd()
#
#     exception_info = ''
#     result = {}
#     try:
#         # Change the temp directories used to build the weave stuff.
#         # Without this the build will fail, due to weave_imp.o being accessed
#         # by multiple processes.
#         if os.path.exists(tempdir_subproc):
#             shutil.rmtree(tempdir_subproc)
#         os.mkdir(tempdir_subproc)
#         tempfile.tempdir = tempdir_subproc
#         os.environ['PYTHONCOMPILED'] = PYTHONCOMPILED_subproc
#     except:
#         traceback.print_exc()
#         exception_info += '\nEXCEPTION SETTING TEMPDIRS: {0}'.format(traceback.format_exc())
#
#     t_start = time.time()
#     try:
#         result = run_task(task_info, sys_params['taskdir'], sys_params['tempdir_subproc'])
#     except:
#         traceback.print_exc()
#         exception_info += '\nEXCEPTION IN RUN: {0}'.format(traceback.format_exc())
#         status = 'error'
#     else:
#         status = 'finished'
#     t_end = time.time()
#     tasktime = t_end - t_start
#
#     os.chdir(cwd)  # Make sure we restore the original CWD
#
#     try:
#         if PYTHONCOMPILED_original:
#             os.environ['PYTHONCOMPILED'] = PYTHONCOMPILED_original
#         elif 'PYTHONCOMPILED' in os.environ:
#             del os.environ['PYTHONCOMPILED']
#     except:
#         traceback.print_exc()
#         exception_info += '\nEXCEPTION RESETTING PYTHONCOMPILED: {0}'.format(traceback.format_exc())
#
#     try:
#         tempfile.tempdir = tempdir_original
#         shutil.rmtree(tempdir_subproc)
#     except:
#         traceback.print_exc()
#         exception_info += '\nEXCEPTION RESETTING TEMPDIR: {0}'.format(traceback.format_exc())
#
#     if exception_info != '':
#         result['log_file'] = {'exc_info': exception_info}
#     simresult = SimulationResult(task_id, result, status, tasktime)
#     try:
#         result_q.put(simresult, block=True)
#     except IOError:
#         traceback.print_exc()
#         simresult.status = 'error_ipc'
#         log_file = simresult.finaldata.setdefault('log_file', {})
#         exc_info = log_file.setdefault('exc_info', '')
#         exc_info += traceback.format_exc()
#         result_q.put(simresult, block=True)
#
#
# def _estimate_time_remaining(single_times, elapsed_time, num_sims, num_procs):
#     import numpy as np
#     num_procs = max(num_procs, 1)
#     avg_wall_time = np.mean(single_times)
#     finished_counter = len(single_times)
#     num_remaining = num_sims - finished_counter
#     fully_parallel = num_remaining / num_procs #integer division
#     partial_parallel = np.minimum(num_remaining % num_procs,1)
#     remaining_est = avg_wall_time*(fully_parallel+partial_parallel)
#
#     return remaining_est
#
#
# def _get_sys_params():
#     import tempfile
#
#     tempdir_orig = tempfile.gettempdir()
#     user_time = str(os.getuid()) + '{0:.6f}'.format(time.time())
#     tempdir_stem = os.path.join(tempdir_orig, user_time)
#
#     sys_params = {'tempdir_original': tempdir_orig, 'tempdir_stem': tempdir_stem}
#
#     if 'PYTHONCOMPILED' in os.environ:
#         PYTHONCOMPILED_original = os.environ['PYTHONCOMPILED']
#     else:
#         PYTHONCOMPILED_original = None
#     sys_params['PYTHONCOMPILED_original'] = PYTHONCOMPILED_original
#     return sys_params
#
#
# def _add_sys_params_for_task(job_info, sys_params, task_id):
#     """
#     Add task specific
#     """
#     sys_params = dict(sys_params)
#
#     tn = job_info.task_name(task_id)
#     sys_params['task_name'] = tn # used to be results_group
#     sys_params['taskdir'] = os.path.join(job_info.job_dir, tn) # used to be subprocdir
#
#     tempdir_stem = sys_params['tempdir_stem']
#     tempdir_subproc = tempdir_stem + tn
#     sys_params['tempdir_subproc'] = tempdir_subproc
#
#     PYTHONCOMPILED_original = sys_params['PYTHONCOMPILED_original']
#     PYTHONCOMPILED_subproc = (tempdir_subproc,PYTHONCOMPILED_original) if PYTHONCOMPILED_original else (tempdir_subproc,)
#     sys_params['PYTHONCOMPILED_subproc'] = (';' if (sys.platform=='win32') else ':').join(PYTHONCOMPILED_subproc)
#
#     return sys_params
#
#
# def _backup_code(job_dir):
#     import shutil
#     code_dir = os.path.join(job_dir, 'code')
#     assert not os.path.exists(code_dir)
#     os.mkdir(code_dir)
#
#     script_file = os.path.join(os.getcwd(),sys.argv[0])
#     shutil.copy2(script_file, code_dir)
#
#     snep_dir = os.path.dirname(os.path.realpath(__file__))
#     ignore = shutil.ignore_patterns('*.pyc', '.git', '.svn', '.idea')
#     shutil.copytree(snep_dir, os.path.join(code_dir,'snep'), ignore=ignore)
#
#
# def _run_remote(job_name, remote_codedir, resultdir, env, mem_per_task, job_time, statepath):
#     from functools import partial
#     numprocs = snep.configuration.config['procs']
#     numslots = numprocs + 1
#
#     on_head = snep.configuration.on_head
#     if not on_head:
#         fab_local("rsync -a * ge:~/{remote_codedir}".format(remote_codedir=remote_codedir))
#
#     poll_interval = -1.
#     while poll_interval < 0:
#         inp = raw_input('Desired polling interval in minutes? [None] ')
#         try: poll_interval = float(inp)
#         except: poll_interval = 0
#
#     get_result = raw_input('Retrieve result? y/[n] ') == 'y' if (not on_head and poll_interval > 0) else False
#     delete_result = 'y' == raw_input('Do you want to delete the remote file? y/[n] ') if (not on_head and get_result) else False
#
#     ts = time.strftime('%Y-%m-%d-%Hh%Mm%Ss')
#     jobdir = '~/{resultdir}/{job_name}/{ts}'.format(resultdir=resultdir, job_name=job_name, ts=ts)
#     cmd = 'qsub -V -q cognition-all.q -cwd  -b y' #
#     cmd += ' -N {job_name} -pe cognition.pe {numslots}'.format(job_name=job_name, numslots=numslots)
#     if mem_per_task:
#         cmd += ' -l h_vmem={vmem}G'.format(vmem=mem_per_task) #*numprocs
#     if job_time:
#         cmd += ' -l h_rt={0}'.format(job_time)
#     cmd += ' -o {jobdir}/stdout -e {jobdir}/stderr'.format(jobdir=jobdir)
#     cmd += ' -v {env} SNEPJOBDIR={jobdir} NUMPROCS={numprocs}'.format(env=env, jobdir=jobdir, numprocs=numprocs)
#     if statepath:
#         cmd += ' STATE={statepath}'.format(statepath=statepath)
#     cmd += ' python {job_name}.py'.format(job_name=job_name) # mpirun -np 1
#
#     if not on_head:
#         qsub = partial(fab_run, cmd)
#         qstat = partial(fab_run, 'qstat')
#         with fab_settings(host_string=snep.configuration.cluster_url):
#             with fab_cd("~/{remote_codedir}".format(remote_codedir=remote_codedir)):
#                 fab_run('mkdir {}'.format(jobdir))
#                 _do_qsub(qsub, qstat, resultdir, job_name, poll_interval, get_result, delete_result, jobdir)
#     else:
#         def qstat():
#             m = fab_local('qstat', capture=True)
#             print(m.stdout)
#             return m
#         def qsub():
#             return fab_local(cmd, capture=True)
#         fab_local('mkdir {}'.format(jobdir))
#         _do_qsub(qsub, qstat, resultdir, job_name, poll_interval, get_result, delete_result, jobdir)
# #         disp = fab_run('echo $DISPLAY')
#     time.sleep(1)
#
#
# def _do_qsub(qsub, qstat, resultdir, job_name, poll_interval, get_result, delete_result, jobdir):
#     import re
#     #jobdir = os.path.join(home, 'experiments', job_name, time.strftime('%Y-%m-%d-%Hh%Mm%Ss'))
#     #cmd += ' --jd {jobdir} --procs {numprocs}'.format(jobdir=jobdir, numprocs=numprocs)
#     start = time.time()
#     res_qsub = qsub()
#     jids = re.findall('[0-9]{6}', str(res_qsub.stdout))
#     assert(len(jids)==1)
#     time.sleep(1)
#     res_qstat = qstat()
#     if poll_interval > 0:
#         job_num = jids[0]
#         while job_num in str(res_qstat.stdout):
#             time.sleep(poll_interval*60)
#             res_qstat = qstat()
#         if get_result:
#             _retrieve_result(job_name, resultdir, jobdir, delete_result)
#         t = time.time() - start
#         print('Job completed at {0} after {1:.1f} minutes!'.format(time.strftime('%m-%d-%Hh%Mm%Ss'), t/60.))
#     else:
#         print('You will have to manually check your job using qstat')
#
#
# def _retrieve_result(job_name, resultdir, jobdir, delete_result):
#     fab_local('scp -r ge:{jobdir} ~/{resultdir}/{job_name}/'.format(
#               jobdir=jobdir, resultdir=resultdir, job_name=job_name))
#     if delete_result:
#         with fab_settings(host_string=snep.configuration.cluster_url):
#             fab_run('rm -rf {jobdir}'.format(jobdir=jobdir))
#         # jobdir = '~/{resultdir}/{job_name}/'.format(resultdir=resultdir, job_name=job_name)
#         # output = fab_run('ls {jobdir}'.format(jobdir=jobdir))
#         # dirs = output.split()
#         # if dirs:
#         #     d = dirs[user_select_from_list(dirs, 'Chose an experiment to download')]
#         #     remotedir = jobdir+d
#         #     fab_local('scp -r ge:{remotedir} ~/{resultdir}/{job_name}/'.format(
#         #             remotedir=remotedir, resultdir=resultdir, job_name=job_name))
#         #     if auto_delete or 'y' == raw_input('Do you want to delete the remote file? y/[n] '):
#         #         fab_run('rm -rf {remotedir}'.format(remotedir=remotedir))
#         # else:
#         #     print('No results found in {jobdir}'.format(jobdir=jobdir))
#     time.sleep(1)  # sleep to avoid exception due to fast interpreter shutdown
#
