# import os
# DIR = '/TRT'
# directories = os.listdir(DIR)
# file = open('command.bat', 'wt')
# for directory in directories:
#     file.write(f'mkdir {DIR}/{directory}-zip\n')
#     file.write(f'cd {DIR}/{directory}-zip\n')
#     file.write(f'gh repo create {directory} --public --description "Reference for {directory}"\n')
#     file.write(f'zip -r -s 20m {directory}.zip ../{directory}\n')
#     file.write('git init\n')
#     file.write('git add .\n')
#     file.write(f'git commit -m "reference {directory}"\n')
#     file.write(f'git config user.name TrevisanoEros\n')
#     file.write(f'git config user.email trevisanoeros@gmail.com\n')
#     file.write(f'git remote add origin https://github.com/TrevisanoEros/{directory}\n')
#     file.write(f'git push -u origin master\n')
#     # file.write(f'TrevisanoEros\n')
#     # file.write(f'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\n')
#     file.write("\n\n")

# file.close()

# import os
# xx = os.listdir('/home/ubuntu/audio/ffhq')
# xx.sort(key=lambda x: int(x.split('.z')[-1]) if not x.endswith('zip') else 0)
# file = open('command.bat', 'wt')
# file.write('cd ffhq\n')
# file.write('git init\n')
# file.write('git config user.name TrevisanoEros\n')
# file.write('git config user.email trevisanoeros@gmail.com\n')
# file.write('git remote add origin https://github.com/TrevisanoEros/ffhq-dataset\n')
# for i, filename in enumerate(xx):
# 	file.write(f'git add {filename}\n')
# 	if i % 100 == 0:
# 		file.write(f'git commit -m "added to {i}"\n')
# 		file.write('git push -u origin master\n\n')
# else:
# 	file.write(f'git commit -m "added to {i}"\n')
# 	file.write('git push -u origin master\n\n')
# file.close()

import pickle
time_array = pickle.load(open('time_array.pkl', 'rb'))
avg_time_array = []
for i, array in enumerate(time_array):
    if array == []:
        avg_time_array.append(0)
    else:
        avg_time_array.append(sum(array) / len(array))
print(avg_time_array)
