import subprocess
import os, sys


def batch_convert_ui_to_py(source_directory, target_directory=None, pyside2_uic_exe: str = None):
    """

    :param source_directory: Location of the ui files
    :param target_directory: Destination of the converted ui-files (defaults to source_directory)
    :param pyside2_uic_exe: Path to pyside2_uic_exe (leave blank if part of environment)
    :return:
    """
    # Check target_directory
    if target_directory is None:
        target_directory = source_directory

    # Check command:
    if pyside2_uic_exe is None:
        pyside2_uic_exe = 'pyside2-uic --from-imports'

    # List files:
    ui_files = [f for f in os.listdir(source_directory) if os.path.splitext(f)[-1] == '.ui']

    # Convert files
    for i, ui_file in enumerate(ui_files):
        # Define file names
        old_fn = '{}/{}'.format(source_directory, ui_file)
        new_fn = '{}/ui_{}.py'.format(target_directory, os.path.splitext(ui_file)[0])
        print('Converting ({2}/{3}): \'{0}\' to \'{1}\'...'.format(old_fn,  new_fn, i+1, len(ui_files)))
        subprocess.call('{0} {1} -o {2}'.format(pyside2_uic_exe, old_fn, new_fn))


if __name__ == "__main__":
    batch_convert_ui_to_py(*sys.argv[1:])
