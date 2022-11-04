import numpy as np
import random
all = ['SN002','SN010','SN001','SN026','SN027','SN032','SN030','SN009','SN016','SN013','SN018','SN011','SN028','SN012','SN006','SN031','SN021','SN024','SN003','SN029','SN023','SN025','SN008','SN005','SN007','SN017','SN004']
random.shuffle(all)
part1 = all[0:9]
part2 = all[9:18]
part3 = all[18:]

print(all)
print(part1)
print(part2)
print(part3)