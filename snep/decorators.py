from functools import wraps


def defaultmonitors(preproc):
    @wraps(preproc)
    def decorated_preproc(task_info, taskdir, tempdir, neuron_groups, local_objects, monitors):
        """
        Simply constructs Brian Monitors as specified by the user.
        """
        from snep.library.brianimports import SpikeMonitor, PopulationRateMonitor, StateMonitor, Clock

        neuron_groups, local_objects, usermons = preproc(task_info, taskdir, tempdir, neuron_groups, local_objects,
                                                         monitors)

        monitor_objs = {'spikes': {}, 'poprate': {}, 'statevar': {}, }

        for montypename, mons in usermons.items():
            monitor_objs[montypename].update(mons)

        if 'spikes' in monitors:
            for popname in monitors['spikes']:
                monitor_objs['spikes'][popname] = SpikeMonitor(neuron_groups[popname])

        if 'poprate' in monitors:
            for popname in monitors['poprate']:
                monitor_objs['poprate'][popname] = PopulationRateMonitor(neuron_groups[popname])

        if 'statevar' in monitors:
            for clock_dt, pop_to_varnames in monitors['statevar'].items():
                clk = Clock(dt=clock_dt.quantity)
                for popname, varnames in pop_to_varnames.items():
                    varnames, record = varnames if isinstance(varnames, tuple) else (varnames, True)
                    if popname not in monitor_objs['statevar']:
                        monitor_objs['statevar'][popname] = {}
                    sm = StateMonitor(neuron_groups[popname], varnames, clock=clk, record=record)
                    if isinstance(varnames, (tuple, list)):
                        for vn in varnames:
                            monitor_objs['statevar'][popname][vn] = sm
                    else:
                        monitor_objs['statevar'][popname][varnames] = sm
                    #         if 'weights' in monitors:
                    #             for (clock_dt, record_stats, record_values, record_last), con_names in monitors['weights'].iteritems():
                    #                 clk = Clock(clock_dt.quantity)
                    #                 for con_name, bins, N_samples in con_names:
                    #                     con = neuron_groups[con_name]
                    #                     wm = WeightMonitor(con, record_stats, record_values, record_last, bins, clk, N_samples)
                    #                     monitor_objs['weights'][con_name] = wm

        return neuron_groups, local_objects, monitor_objs

    return decorated_preproc

# This decorator is no longer used, since we do not write the results 
# in the worker process anymore.
# def savedefaultmonitors(saverates,savespikes,savestates):
#    def decoratormaker(postproc):
#        @wraps(postproc)
#        def decorated_postproc(params, rawdata, resultsfile=None):
#            from snep.library.tables.results import SubprocessResultsTables
#            local_results = not resultsfile
#            if local_results:
#                resultsfile = SubprocessResultsTables(params['results_file'],
#                                                      params['results_group'])
#                resultsfile.open_file()
#                resultsfile.initialize()
#    
#            if saverates and 'poprate' in rawdata:
#                for pop_name, (times, rates) in rawdata['poprate'].iteritems():
#                    resultsfile.add_population_rates(pop_name, times, rates)
#            if savespikes and 'spikes' in rawdata:
#                for pop_name, spikes in rawdata['spikes'].iteritems():
#                    resultsfile.add_spiketimes(pop_name, spikes)
#            if savestates and 'statevar' in rawdata:
#                for pop_name, all_vars in rawdata['statevar'].iteritems():
#                    for varname, (times, values) in all_vars.iteritems():
#                        resultsfile.add_state_variables(pop_name, varname, times, values)
#        
#            postproc_res = postproc(params, rawdata, resultsfile)
#            if local_results:
#                resultsfile.close_file()
#            return postproc_res
#        return decorated_postproc
#    return decoratormaker
