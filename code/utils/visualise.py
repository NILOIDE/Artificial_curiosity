from tensorboardX import SummaryWriter
from datetime import datetime
import csv
import os


class Visualise:
    def __init__(self, **kwargs):
        super(Visualise, self).__init__()

        self.vis_args = kwargs
        self.time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        self.folder_name = kwargs['save_dir']+self.time_stamp+'_-_'+kwargs['name']
        self.writer = SummaryWriter(self.folder_name)
        self.x_interval = kwargs['interval']
        self.train_id = self.x_interval  # train count for visualisation
        self.eval_id = 0  # valid count for visualisation
        # os.mkdir('results/' + kwargs['name'])
        with open(self.folder_name + '/params.csv', mode='w') as f:
            csv_writer = csv.writer(f)
            for key, value in kwargs.items():
                csv_writer.writerow([key, value])

    def train_iteration_update(self, **kwargs):
        self.writer.add_scalar("Training/Mean ep extrinsic rewards", kwargs['ext'], self.train_id)
        self.writer.add_scalar("Training/Mean step intrinsic rewards", kwargs['int'], self.train_id)
        self.writer.add_scalar("Training/WM loss", kwargs['wm_loss'], self.train_id)
        if 'wm_t_loss' in kwargs:
            self.writer.add_scalar("Training/WM translation loss", kwargs['wm_t_loss'], self.train_id)
        if 'wm_ng_loss' in kwargs:
            self.writer.add_scalar("Training/WM negative sampling loss", kwargs['wm_ng_loss'], self.train_id)
        self.writer.add_scalar("Training/Policy loss", kwargs['alg_loss'], self.train_id)
        self.train_id += self.x_interval

    def eval_iteration_update(self, **kwargs):
        self.writer.add_scalar("Evaluation/Mean ep extrinsic rewards", kwargs['ext'], self.eval_id)
        self.writer.add_scalar("Evaluation/Mean step intrinsic rewards", kwargs['int'], self.eval_id)
        self.writer.add_scalar("Evaluation/WM loss", kwargs['wm_loss'], self.eval_id)
        self.writer.add_scalar("Evaluation/Policy loss", kwargs['alg_loss'], self.eval_id)
        self.eval_id += self.x_interval
