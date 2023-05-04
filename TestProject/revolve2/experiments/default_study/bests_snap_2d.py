import os
import cv2
import pprint
import math
import numpy as np


async def main() -> None:

    study = 'default_study'
    experiments_name = ['speed', 'purespeed']
    runs = list(range(2, 20+1))
    generations = [200]
    bests = 5
    path_out = f'/storage/karine/{study}/analysis/snapshots'

    for gen in generations:
        # TODO: change black background to white
        for experiment_name in experiments_name:
            horizontal = []
            print(experiment_name)
            for run in runs:
                print('  run: ', run)

                print('  gen: ', gen)

                if not os.path.exists(path_out):
                        os.makedirs(path_out)
                path_in = f'{path_out}/{experiment_name}/run_{run}/gen_{gen}'
                lst = os.listdir(path_in)
                lst = lst[0:bests]
                print(lst)
                for_concats = [cv2.imread(f'{path_in}/{robot}') for robot in lst]
                heights = [o.shape[0] for o in for_concats]
                max_height = max(heights)
                margin = 20

                for idx, c in enumerate(for_concats):
                    if for_concats[idx].shape[0] < max_height:
                        bottom = max_height - for_concats[idx].shape[0] + margin
                    else:
                        bottom = margin

                    for_concats[idx] = cv2.copyMakeBorder(for_concats[idx], margin, math.ceil(bottom), margin,\
                                                           margin, cv2.BORDER_CONSTANT, None, value=[0, 0, 0]) #value=[255, 255, 255])
                    #for_concats[idx][np.where((for_concats[idx] == [0, 0, 0]).all(axis=2))] = [255, 255, 255]

                concats = cv2.hconcat(for_concats)
                horizontal.append(concats)
                #concats[np.where((concats == [0, 0, 0]).all(axis=2))] = [255, 255, 255]

            widths = [o.shape[1] for o in horizontal]

            max_width = max(widths)
            for idx, img in enumerate(horizontal):
                if horizontal[idx].shape[1] < max_width:
                    right = max_width - horizontal[idx].shape[1]
                else:
                    right = 0

                horizontal[idx] = cv2.copyMakeBorder(horizontal[idx], 0, margin*6, 0,\
                                                       math.ceil(right), cv2.BORDER_CONSTANT, None, value=3)


            vertical = cv2.vconcat(horizontal)
            #vertical[np.where((vertical == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
            
            cv2.imwrite(f'{path_out}/bests_{experiment_name}_{gen}.png', vertical)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

# can be run from root