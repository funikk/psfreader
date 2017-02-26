import psfreader
import psfreader.psfdata as psfdata
import numpy as np
import click


@click.command()
@click.argument('psffile')
@click.argument('csvfile')
def read_write(psffile, csvfile):
    print('reading psffile: ' + psffile)
    r = psfreader.PSFReader(psffile)
    with open(csvfile, 'w', encoding='utf-8') as f:
        if not r.is_swept():
            print('Does not support a PSF file without sweep.')
            return None
        signals = r.get_signal_names()
        s_types = [r.get_signal_types(name) for name in signals]
        vectors = [r.get_signal(s) for s in signals]
        sweep = r.get_sweep_values()
        sweep_title = r.get_sweep_param_name()
        
        t_set = set(s_types)
        if not t_set.issubset({psfdata.TypeId.DOUBLE, psfdata.TypeId.COMPLEX_DOUBLE}):
            print('Unsupported data types.')
            return None
        
        length = r.get_sweep_npoints()
        
        f.write('# psf2tsv sample\n')
        f.write(sweep_title)
        for (s, t) in zip(signals, s_types):
            f.write('\t')
            if t == psfdata.TypeId.COMPLEX_DOUBLE:
                f.write('REAL:' + s)
                f.write(' IMAG:' + s)
            else:
                f.write(s)
        f.write('\n')
        
        for i in range(length):
            f.write('{:E}'.format(sweep[i]))
            for (v, t) in zip(vectors, s_types):
                f.write('\t')
                if t == psfdata.TypeId.COMPLEX_DOUBLE:
                    f.write('{:E} {:E}'.format(v[i].real, v[i].imag))
                else:
                    f.write('{:E}'.format(v[i]))
            f.write('\n')
    print('Done!')

if __name__ == '__main__':
    read_write()
