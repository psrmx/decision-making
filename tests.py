import unittest
import numpy as np
from helper_funcs import validate_file, handle_brian_units
from burst_analysis import spks2neurometric
from brian2.units import second, Hz, pA, ms


class HelperFuncsTest(unittest.TestCase):

    def test_validate_file_out_of_scope_raise_exception(self):
        bads = (2, None, {'a': 'filename'}, ['a_filename'])
        for bad_input in bads:
            with self.assertRaises(ValueError) as m:
                validate_file(bad_input)
                if not hasattr(m, 'exception'):
                    self.fail('{} as filename did not raise exception'.format(bad_input))

    def test_handle_brian_units(self):
        self.assertEqual(handle_brian_units('80*pA'), (80, 'pA'))
        self.assertEqual(handle_brian_units('-100*Hz'), (-100, 'Hz'))
        self.assertEqual(handle_brian_units('5'), 5)
        self.assertEqual(handle_brian_units('False'), None)
        self.assertEqual(handle_brian_units('kk'), None)

    def test_burst_analysis_correct_isis(self):
        spks2test = np.loadtxt('spksSE.csv')
        spks2test = spks2test[np.newaxis, :]
        task_info = {
            'dec': {
                'N_E': 1600,
                'N_I': 400,
                'sub': 0.15},

            'sen': {
                'N_E': 1600,
                'N_I': 400,
                'N_X': 1000,
                'sub': 0.5},

            'sim': {
                'sim_dt': 0.1*ms,
                'stim_dt': 1*ms,
                'runtime': 3*second,
                'settle_time': 0*second,
                'stim_on': 0.5*second,
                'stim_off': 2.5*second,
                'replicate_stim': False,
                'num_method': 'euler',
                'seed_con': 1284,
                'smooth_win': 100*ms,
                'valid_burst': 16e-3,
                '2c_model': True,
                'plt_fig1': False,
                'burst_analysis': True,
                'plasticity': False},

            'plastic': {
                # 'targetB': Parameter(2, 'Hz'),
                'tauB': 50000*ms,
                'tau_update': 10*ms,
                'eta0': 5*pA,
                'min_burst_stop': 0.1,
                'dec_winner_rate': 35*Hz},

            'c': 0,
            'bfb': 0,
            'targetB': 2*Hz}

        self.assertEqual(spks2neurometric(task_info, spks2test, raster=True), None)

    def test_brian2param(self):
        pass

    def test_brian_to_parameter(self):
        pass


if __name__ == '__main__':
    unittest.main()
