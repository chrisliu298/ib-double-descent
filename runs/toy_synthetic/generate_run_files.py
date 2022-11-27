import copy
import json

if __name__ == '__main__':
    with open("./toy_synthetic_template.json", "r") as f:
        template = json.load(f)
    # we change dataset, width, and loss for these experiments
    for dataset in ('mnist', 'fashionmnist'):
        for loss in ('mse', 'ce'):
            for width in range(4, 401, 4):
                run = copy.deepcopy(template)
                run['dataset'] = dataset
                run['loss'] = loss
                layer_dims = f"784x{width}x2"
                run['layer_dims'] = layer_dims
                run['work_dir'] = f"/data/users/ibdd/experiments/toy_synthetic/{dataset}/{loss}/layers_{layer_dims}/"
                with open(f"./definitions/{dataset}_{loss}_{layer_dims}.json", "w") as f:
                    json.dump(run, f, sort_keys=True, indent=4)
