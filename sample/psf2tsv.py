import psfreader
import psfreader.psfdata as psfdata
import numpy as np
import argparse

def read_write(psffile, csvfile):
    print('reading psffile: ' + psffile)
    r = psfreader.PSFReader(psffile)
    with open(csvfile, 'w', encoding='utf-8') as f:
        if not r.is_swept():
            print('Does not support a PSF file without sweep.')
            return None
        signals = r.get_signal_names()
        s_types = [r.get_signal_types(name) for name in signals]
        units = [r.get_signal_units(name) for name in signals]
        vectors = [r.get_signal(s) for s in signals]
        sweep = r.get_sweep_values()
        sweep_title = r.get_sweep_param_name()
        
        t_set = set(s_types)
        if not t_set.issubset({psfdata.TypeId.DOUBLE, psfdata.TypeId.COMPLEX_DOUBLE}):
            print('Unsupported data types.')
            return None
        
        length = r.get_sweep_npoints()
        length2 = r.get_read_npoints()
        
        if length != length2:
            print('Data length is mismatched. Target file is broken or incompleted!')
            length = length2
        
        f.write('# psf2tsv sample\n')
        f.write(sweep_title)
        for (s, t, u) in zip(signals, s_types, units):
            if u == 'V':
                n = 'v({})'.format(s)
            elif u == 'A':
                n = 'i({})'.format(s)
            else:
                n = s

            f.write('\t')
            if t == psfdata.TypeId.COMPLEX_DOUBLE:
                f.write('REAL:' + n)
                f.write('\tIMAG:' + n)
            else:
                f.write(n)
        f.write('\n')
        
        for i in range(length):
            f.write('{:E}'.format(sweep[i]))
            for (v, t) in zip(vectors, s_types):
                f.write('\t')
                if t == psfdata.TypeId.COMPLEX_DOUBLE:
                    f.write('{:E}\t{:E}'.format(v[i].real, v[i].imag))
                else:
                    f.write('{:E}'.format(v[i]))
            f.write('\n')
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('psffile')
    parser.add_argument('csvfile')
    opt = parser.parse_args()
    
    read_write(opt.psffile, opt.csvfile)
