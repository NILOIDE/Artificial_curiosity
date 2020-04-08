from tensorboardX import SummaryWriter
import numpy as np
import csv
import os


class Visualise:
    def __init__(self, run_name, **kwargs):
        super(Visualise, self).__init__()

        self.vis_args = kwargs
        self.folder_name = run_name
        self.writer = SummaryWriter(self.folder_name)
        self.x_interval = kwargs['interval']
        self.train_id = self.x_interval  # train count for visualisation
        self.eval_id = 0  # valid count for visualisation
        # os.mkdir('results/' + kwargs['name'])
        with open(f'{self.folder_name}/params{kwargs["time_stamp"]}.csv', mode='w') as f:
            csv_writer = csv.writer(f)
            for key, value in kwargs.items():
                csv_writer.writerow([key, value])
        # self.train_iteration_update(ext=0.0, int=0.0, wm_loss=0.0, alg_loss=0.0)

    @staticmethod
    def normalize(img: np.ndarray) -> np.ndarray:
        assert len(img.shape) == 1 or len(img.shape) == 3, 'Image must have channel dimension'
        if img.shape[0] == 1:
            min = np.amin(img, axis=(1, 2))
            max = np.amax(img, axis=(1, 2))
        elif img.shape[0] == 3:
            min = np.amin(img, axis=(1, 2))[:, np.newaxis, np.newaxis]
            max = np.amax(img, axis=(1, 2))[:, np.newaxis, np.newaxis]
        else:
            raise AssertionError(f'Image must be Grayscale or RGB, but found channel dim was of size {img.shape[0]}')
        # return np.sqrt((img - min) / (max - min + 1e-8))
        return (img / (max + 1e-8))

    def train_iteration_update(self, t=None, **kwargs):
        if t is None:
            t = self.train_id
            self.train_id += self.x_interval
            self.writer.add_scalar("Training/WM loss", kwargs['wm_loss'], t)
            if 'wm_t_loss' in kwargs:
                self.writer.add_scalar("Training/WM translation loss", kwargs['wm_t_loss'], t)
            if 'wm_ng_loss' in kwargs:
                self.writer.add_scalar("Training/WM negative sampling loss", kwargs['wm_ng_loss'], t)
            self.writer.add_scalar("Training/Policy loss", kwargs['alg_loss'], t)

        self.writer.add_scalar("Training/Mean ep extrinsic rewards", kwargs['ext'], t)
        self.writer.add_scalar("Training/Mean step intrinsic rewards", kwargs['int'], t)

    def eval_iteration_update(self, **kwargs):
        # self.writer.add_scalar("Evaluation/Mean ep extrinsic rewards", kwargs['ext'], self.eval_id)
        # self.writer.add_scalar("Evaluation/Mean step intrinsic rewards", kwargs['int'], self.eval_id)
        # self.writer.add_scalar("Evaluation/WM loss", kwargs['wm_loss'], self.eval_id)
        # self.writer.add_scalar("Evaluation/Policy loss", kwargs['alg_loss'], self.eval_id)
        density_map, pe_map, ape_map = None, None, None
        if 'density_map' in kwargs:
            if len(kwargs['density_map'].shape) == 2:
                kwargs['density_map'] = np.expand_dims(kwargs['density_map'], axis=0)
            density_map = self.normalize(kwargs['density_map'])
            if 'walls_map' in kwargs and kwargs['walls_map'] is not None:
                density_map_rgb = np.concatenate((density_map,
                                                  density_map + kwargs['walls_map'],
                                                  density_map), axis=0)
                self.writer.add_image("Evaluation/Visitation densities", density_map_rgb, self.eval_id)
            else:
                self.writer.add_image("Evaluation/Visitation densities", density_map, self.eval_id)
        if 'pe_map' in kwargs:
            if len(kwargs['pe_map'].shape) == 2:
                kwargs['pe_map'] = np.expand_dims(kwargs['pe_map'], axis=0)
            # Remove nodes that are walls
            if 'walls_map' in kwargs and kwargs['walls_map'] is not None:
                kwargs['pe_map'] = kwargs['pe_map'] * (1 - kwargs['walls_map'])
            # Add scalar before normalising
            self.writer.add_scalar("Evaluation/Prediction error map sum", kwargs['pe_map'].sum(), self.eval_id)
            pe_map = self.normalize(kwargs['pe_map'])
            # Display walls as red if map walls are given by converting error map to RGB
            if 'walls_map' in kwargs and kwargs['walls_map'] is not None:
                pe_map_rgb = np.concatenate((pe_map,
                                             pe_map + kwargs['walls_map'],
                                             pe_map), axis=0)
                self.writer.add_image("Evaluation/Prediction error map", pe_map_rgb, self.eval_id)
            else:
                self.writer.add_image("Evaluation/Prediction error map", pe_map, self.eval_id)
            # Density-Error RGB mix
            if density_map is not None:
                overlap = np.concatenate((pe_map,
                                          np.zeros_like(density_map) + kwargs['walls_map'],
                                          density_map), axis=0)
                self.writer.add_image("Evaluation/Density and Prediction error overlap", overlap, self.eval_id)
        if 'ape_map' in kwargs and kwargs['ape_map'] is not None:
            if len(kwargs['ape_map'].shape) == 2:
                kwargs['ape_map'] = np.expand_dims(kwargs['ape_map'], axis=0)
            # Remove nodes that are walls
            if 'walls_map' in kwargs and kwargs['walls_map'] is not None:
                kwargs['ape_map'] = kwargs['ape_map'] * (1 - kwargs['walls_map'])
            # Add scalar before normalising
            self.writer.add_scalar("Evaluation/Argmax prediction error map sum", kwargs['ape_map'].sum(), self.eval_id)
            ape_map = self.normalize(kwargs['ape_map'])
            # Display walls as red if map walls are given by converting error map to RGB
            if 'walls_map' in kwargs and kwargs['walls_map'] is not None:
                ape_map_rgb = np.concatenate((ape_map,
                                              ape_map + kwargs['walls_map'],
                                              ape_map), axis=0)
                self.writer.add_image("Evaluation/Argmax prediction error map", ape_map_rgb, self.eval_id)
            else:
                self.writer.add_image("Evaluation/Argmax prediction error map", ape_map, self.eval_id)
        self.eval_id += self.x_interval

    def close(self):
        self.writer.close()
