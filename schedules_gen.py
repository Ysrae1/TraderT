
from itertools import permutations
import os

proportions2 = [
    [4, 4, 4, 4, 4],
    [8, 4, 4, 4, 0],
    [8, 8, 4, 0, 0],
    [16, 4, 0, 0, 0],
    [20, 0, 0, 0, 0],
    [10, 5, 5, 0, 0],
    [10, 10, 0, 0, 0],
]

proportions = [
    [5, 5, 5, 5, 0],
    [8, 4, 4, 4, 0],
    [8, 8, 2, 2, 0],
    [10, 4, 4, 2, 0],
    [12, 4, 2, 2, 0],
    [14, 2, 2, 2, 0],
    [16, 2, 2, 0, 0],
    [16, 4, 0, 0, 0],
    [18, 2, 0, 0, 0],
    [20, 0, 0, 0, 0],
]
perms = [p for i in proportions for p in set(permutations(i))]
to_append = [0, 0]

# open csv file to write
output_dir = './simulator/markets'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

rows_per_section = 5
num_sections = len(perms)//rows_per_section
for i in range(num_sections):
    start_index = i * rows_per_section
    perm_s = perms[start_index:start_index + rows_per_section]
    
    with open(f"{output_dir+'/'}{str(i).zfill(len(str(num_sections - 1)))}.csv","w") as f:
        for perm in perm_s:
            f.write(
                f"{perm[0]},{perm[1]},{perm[2]},{perm[3]},{perm[4]},{to_append[0]},{to_append[1]}\n"
            )

