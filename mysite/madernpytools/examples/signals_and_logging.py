import traitlets, threading, time
import madernpytools.log as mlog
import madernpytools.signal_handling as msigs
import numpy as np
from madernpytools.signal_handling import ISignalKeyList


class SomeGenerator(traitlets.HasTraits, msigs.ISignalProvider):   # Indicates that we implement traits in this class
    output = traitlets.TraitType()            # Our output signal

    def __init__(self, rate=1):
        super().__init__(output={}  # Initialize 'output'
                         )   # Initialize base classe

        self._rate = rate
        self._keep_running = True

    # Following are required by our signal provider interface:
    @property
    def required_input_keys(self) -> ISignalKeyList:
        """Returns the keys which are required by this class (returns none, because we implemented a generator

        :return:
        """
        return msigs.SignalKeyList([])

    @property
    def added_keys(self) -> ISignalKeyList:
        """Returns 'x, y z'

        :return:
        """
        return msigs.SignalKeyList(['x', 'y', 'z'])

    def start_generation(self):
        """ Start data generation

        :return:
        """
        self._keep_running = True
        self._thrd = threading.Thread(target=self._worker)
        self._thrd.start()

    @property
    def sampling_rate(self):
        return self._rate

    def stop_generation(self):
        """ Stop data generation

        :return:
        """
        self._keep_running = False

    def _worker(self):
        while self._keep_running:
            self.output = dict(zip(['x', 'y', 'z'], np.random.randn(3)))
            time.sleep(self._rate**-1)


if __name__=="__main__":

    generator = SomeGenerator(rate=10)                                     # Dummy generator

    # Create new log
    # Log info:
    log_info = mlog.LogInfo(description='Test Log',
                            signal_header=generator.added_keys,
                            sampling_rate=generator.sampling_rate
                            )
    # Define log:
    log = mlog.Log(log_info)

    # Link generator to log:
    traitlets.link((generator, 'output'), (log, 'input'))

    # Activate Generation
    generator.start_generation()

    # Activate log
    log.active = True

    # Wait for some time
    for i in range(10):
        print(f'Logged {log.n_samples} samples')
        time.sleep(1)

    # Write log:
    log.active = False
    log.save('test_log.csv')

