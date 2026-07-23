import runpy
import sys

sys.path.append(r"D:\anaconda\Lib\site-packages")
script = sys.argv.pop(1)
runpy.run_path(script, run_name="__main__")
