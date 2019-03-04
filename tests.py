import unittest
from helper_funcs import validate_file, handle_brian_units


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

    def test_brian2param(self):
        pass

    def test_brian_to_parameter(self):
        pass


if __name__ == '__main__':
    unittest.main()
