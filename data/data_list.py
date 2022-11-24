import json
import os
def load_file_list_recursion(fpath, result):
    allfilelist = os.listdir(fpath)
    for file in allfilelist:
        filepath = os.path.join(fpath, file)
        if os.path.isdir(filepath):
            load_file_list_recursion(filepath, result)
        else:
            result.append(filepath)
            print(len(result))



def scan(input_path, out_put):
    result_list = []
    load_file_list_recursion(input_path, result_list)
    result_list.sort()

    for i in range(len(result_list)):
        print('{}_{}'.format(i, result_list[i]))

    with open(out_put, 'w') as j:
        json.dump(result_list, j)

scan('CelebA/train', './data/train_flist.txt')
scan('CelebA/valid', './data/valid_flist.txt')
scan('CelebA/test', './data/test_flist.txt')
scan('CelebA/mask-train', './data/mask-train_flist.txt')
scan('CelebA/mask-valid', './data/mask-valid_flist.txt')
scan('CelebA/mask-test', './data/mask-test_flist.txt')
