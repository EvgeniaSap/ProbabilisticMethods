import os

pathFrom = './Design_ui'
pathTo = './Design_py'
files = [i for i in os.listdir(pathFrom) if i.endswith('.ui')]
for f in files:
    com = f'pyuic5 {pathFrom}/{f} -o {pathTo}/{f[:-3]}.py'
    os.system(com)