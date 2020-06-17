from tensorboardX import SummaryWriter
import numpy as np
import csv
import sys

np.set_printoptions(threshold=sys.maxsize)


class Visualise:
    def __init__(self, run_name, **kwargs):
        super(Visualise, self).__init__()

        self.vis_args = kwargs
        self.folder_name = run_name
        self.writer = SummaryWriter(self.folder_name)
        self.train_interval = kwargs['export_interval']
        self.eval_interval = kwargs['eval_interval']
        self.train_id = self.train_interval  # train count for visualisation
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
        # return np.sqrt(img / (max + 1e-8))
        new_img = img / max if max != 0.0 else img
        return new_img

    def train_iteration_update(self, t=None, **kwargs):
        if t is None:
            t = self.train_id
            self.train_id += self.train_interval
            self.writer.add_scalar("Training/WM loss", kwargs['wm_loss'], t)
            if 'wm_t_loss' in kwargs:
                self.writer.add_scalar("Training/WM translation loss", kwargs['wm_trans_loss'], t)
            if 'wm_ns_loss' in kwargs:
                self.writer.add_scalar("Training/WM negative sampling loss", kwargs['wm_ns_loss'], t)
            if 'wm_inv_loss' in kwargs:
                self.writer.add_scalar("Training/WM inverse model loss", kwargs['wm_inv_loss'], t)
            if 'wm_vae_loss' in kwargs:
                self.writer.add_scalar("Training/WM VAE loss", kwargs['wm_vae_loss'], t)
            self.writer.add_scalar("Training/Policy loss", kwargs['alg_loss'], t)
            if 'info' in kwargs:
                if 'unique_states' in kwargs['info']:
                    self.writer.add_scalar("Training/Unique states visited",
                                           kwargs['info']['unique_states'], self.eval_id)
                if 'unique_states_percent' in kwargs['info']:
                    self.writer.add_scalar("Training/Percent of states visited",
                                           kwargs['info']['unique_states_percent'], self.eval_id)
                if 'kl' in kwargs['info']:
                    self.writer.add_scalar("Training/Policy-Uniform KL",
                                           kwargs['info']['kl'], self.eval_id)
                if 'uniform_diff' in kwargs['info']:
                    self.writer.add_scalar("Training/Policy-Uniform difference",
                                           kwargs['info']['uniform_diff'], self.eval_id)
                if 'uniform_diff_visited' in kwargs['info']:
                    self.writer.add_scalar("Training/Policy-Uniform difference visited states",
                                           kwargs['info']['uniform_diff_visited'], self.eval_id)

        self.writer.add_scalar("Training/Mean ep extrinsic rewards", kwargs['ext'], t)
        self.writer.add_scalar("Training/Mean step intrinsic rewards", kwargs['int'], t)

    def eval_iteration_update(self, ext, int):
        self.writer.add_scalar("Evaluation/Mean ep extrinsic rewards", ext, self.eval_id)
        self.writer.add_scalar("Evaluation/Mean ep intrinsic rewards", int, self.eval_id)
        self.eval_id += self.eval_interval

    def eval_gridworld_iteration_update(self, **kwargs):
        density_map, pe_map, q_map = None, None, None
        if 'density_map' in kwargs:
            if len(kwargs['density_map'].shape) == 2:
                kwargs['density_map'] = np.expand_dims(kwargs['density_map'], axis=0)
            # Clipped
            density_map = self.normalize(kwargs['density_map'].clip(max=0.01))
            if 'walls_map' in kwargs and kwargs['walls_map'] is not None:
                density_map_rgb = np.concatenate((density_map,
                                                  density_map + kwargs['walls_map'],
                                                  density_map), axis=0)
                self.writer.add_image("Evaluation/Visitation densities (clipped)", density_map_rgb, self.eval_id)
            else:
                self.writer.add_image("Evaluation/Visitation densities (clipped)", density_map, self.eval_id)
            # Not clipped
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
            if density_map is not None and kwargs['walls_map'] is not None:
                overlap = np.concatenate((pe_map,
                                          np.zeros_like(density_map) + kwargs['walls_map'],
                                          density_map), axis=0)
                self.writer.add_image("Evaluation/Density and Prediction error overlap", overlap, self.eval_id)
        if 'q_map' in kwargs and kwargs['q_map'] is not None:
            if len(kwargs['q_map'].shape) == 2:
                kwargs['q_map'] = np.expand_dims(kwargs['q_map'], axis=0)
            # Remove nodes that are walls
            if 'walls_map' in kwargs and kwargs['walls_map'] is not None:
                kwargs['q_map'] = kwargs['q_map'] * (1 - kwargs['walls_map'])
            # Add scalar before normalising
            self.writer.add_scalar("Evaluation/Argmax Q-value map sum", kwargs['q_map'].sum(), self.eval_id)
            q_map = self.normalize(kwargs['q_map'])
            # Display walls as red if map walls are given by converting error map to RGB
            if 'walls_map' in kwargs and kwargs['walls_map'] is not None:
                q_map_rgb = np.concatenate((q_map,
                                            q_map + kwargs['walls_map'],
                                            q_map), axis=0)
                self.writer.add_image("Evaluation/Argmax Q-value map", q_map_rgb, self.eval_id)
            else:
                self.writer.add_image("Evaluation/Argmax Q-value map", q_map, self.eval_id)
        self.eval_id += self.eval_interval

    def eval_wm_warmup(self, t, **kwargs):
        assert t <= 0
        self.writer.add_scalar("Training/WM loss", kwargs['wm_loss'], t)
        if 'wm_t_loss' in kwargs:
            self.writer.add_scalar("Training/WM translation loss", kwargs['wm_trans_loss'], t)
        if 'wm_ns_loss' in kwargs:
            self.writer.add_scalar("Training/WM negative sampling loss", kwargs['wm_ns_loss'], t)
        if 'wm_inv_loss' in kwargs:
            self.writer.add_scalar("Training/WM inverse model loss", kwargs['wm_inv_loss'], t)
        if 'wm_vae_loss' in kwargs:
            self.writer.add_scalar("Training/WM VAE loss", kwargs['wm_vae_loss'], t)

    def close(self):
        self.writer.close()
