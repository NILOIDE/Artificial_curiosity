import imageio
import os

EXT = '.png'


def create_animation(filenames, save_path):
    save_path = save_path + 'density.gif'
    print("Saving gif to: " + save_path)
    with imageio.get_writer(save_path, mode='I') as writer:
        for i, filename in enumerate(filenames[0:-1:2]):
            print(f'{i}/{len(filenames)}', end='\r')
            image = imageio.imread(filename)
            writer.append_data(image)


def all_folders():
    parent = 'final_results/' + 'GridWorldSpiral28x28-v0/'
    for path in [parent + i + '/heat_maps/' for i in os.listdir(parent)]:
        print(path)
        if not os.path.isdir(path):
            print('Does not exist.')
            continue
        if os.path.exists(path[:-len('heat_maps/')]+'density.gif'):
            continue
        files = [i for i in os.listdir(path) if i.endswith(EXT)]
        if len(files) < 5:
            continue
        keys = [int(i[:-len(EXT)]) for i in files]
        sorted_files = [i for _, i in sorted(zip(keys, files))]
        sorted_paths = [path + i for i in sorted_files]
        create_animation(sorted_paths, path[:-len('heat_maps/')])


if __name__ == "__main__":
    all_folders()