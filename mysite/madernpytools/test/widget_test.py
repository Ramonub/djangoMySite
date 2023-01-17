import madernpytools.madern_widgets as mwid
import madernpytools.models.toolset_model as mts
import sys

import logging

_logger = logging.getLogger(f'madernpytools.{__name__}')

if __name__ == "__main__":
    print(sys.version_info)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
    for log_name in ['madernpytools']: #, 'ScanControl']:
        log = logging.getLogger(log_name)
        log.setLevel(logging.INFO)
        log.addHandler(stream)

    _logger.info('Loading Library..')
    lib = mts.MadernModelLibrary().load('../data/library/')

parts_output, parts_widgets = mwid.generate_editors_in_tab([
                                                            mts.RiRoTensionScrew,
                                                            ],
                                                           lib=lib)

