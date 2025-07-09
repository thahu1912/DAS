import csv
import datetime
import os
import pickle as pkl
import wandb
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter

from utilities.misc import gimme_save_string



class CSVWriter:
    """
    WRITE TO CSV FILE
    """

    def __init__(self, save_path):
        self.save_path = save_path
        self.written = []
        self.n_written_lines = {}

    def log(self, group, segments, content):
        if group not in self.n_written_lines.keys():
            self.n_written_lines[group] = 0

        with open(self.save_path + '_' + group + '.csv', "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            if group not in self.written:
                writer.writerow(segments)
            for line in content:
                writer.writerow(line)
                self.n_written_lines[group] += 1

        self.written.append(group)


class InfoPlotter:
    """
    PLOT SUMMARY IMAGE
    """

    def __init__(self, save_path, title='Training Log', figsize=(25, 19)):
        self.save_path = save_path
        self.title = title
        self.figsize = figsize
        self.colors = ['r', 'g', 'b', 'y', 'm', 'c', 'orange', 'darkgreen', 'lightblue']
        self.ov_title = None

    def make_plot(self, base_title, title_append, sub_plots, sub_plots_data):
        sub_plots = list(sub_plots)
        if 'epochs' not in sub_plots:
            x_data = range(len(sub_plots_data[0]))
        else:
            x_data = range(sub_plots_data[np.where(np.array(sub_plots) == 'epochs')[0][0]][-1] + 1)

        self.ov_title = [(sub_plot, sub_plot_data) for sub_plot, sub_plot_data in zip(sub_plots, sub_plots_data) if
                         sub_plot not in ['epoch', 'epochs', 'time']]
        self.ov_title = [(x[0], np.max(x[1])) if 'loss' not in x[0] else (x[0], np.min(x[1])) for x in self.ov_title]
        self.ov_title = title_append + ': ' + '  |  '.join('{0}: {1:.4f}'.format(x[0], x[1]) for x in self.ov_title)
        sub_plots_data = [x for x, y in zip(sub_plots_data, sub_plots)]
        sub_plots = [x for x in sub_plots]

        plt.style.use('ggplot')
        f, ax = plt.subplots(1)
        ax.set_title(self.ov_title, fontsize=22)
        for i, (data, title) in enumerate(zip(sub_plots_data, sub_plots)):
            ax.plot(x_data, data, '-{}'.format(self.colors[i]), linewidth=1.7, label=base_title + ' ' + title)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.tick_params(axis='both', which='minor', labelsize=18)
        ax.legend(loc=2, prop={'size': 16})
        f.set_size_inches(self.figsize[0], self.figsize[1])
        f.savefig(self.save_path + '_' + title_append + '.svg')
        plt.close()


def get_time_string():
    date = datetime.datetime.now()
    return '{}-{}-{}-{}-{}-{}'.format(date.year, date.month, date.day, date.hour, date.minute, date.second)


def init_logging(opt):
    """
    GENERATE LOGGING FOLDER/FILES
    """
    check_folder = opt.save_path + '/' + opt.savename
    if opt.savename == '':
        check_folder = opt.save_path + '/{}_{}_'.format(opt.dataset.upper(), opt.arch.upper()) + get_time_string()
    counter = 1
    while os.path.exists(check_folder):
        check_folder = opt.save_path + '/' + opt.savename + '_' + str(counter)
        counter += 1
    os.makedirs(check_folder)

    if 'experiment' in vars(opt):
        import argparse
        save_opt = {key: item for key, item in vars(opt).items() if key != 'experiment'}
        save_opt = argparse.Namespace(**save_opt)
    else:
        save_opt = opt

    with open(check_folder + '/ParameterInfo.txt', 'w') as f:
        f.write(gimme_save_string(save_opt))
    pkl.dump(save_opt, open(check_folder + "/HyperParameter.pkl", "wb"))

    return check_folder


class ProgressSaver:
    def __init__(self):
        self.groups = {}

    def log(self, segment, content, group=None):
        if group is None:
            group = segment
        if group not in self.groups.keys():
            self.groups[group] = {}

        if segment not in self.groups[group].keys():
            self.groups[group][segment] = {'content': [], 'saved_idx': 0}

        self.groups[group][segment]['content'].append(content)


class LOGGER:
    def __init__(self, opt, sub_loggers=None, prefix=None, start_new=True):
        """
        LOGGER Internal Structure:

        self.progress_saver: Contains multiple ProgressSaver instances to log metric for scripts metric subsets
                            (e.g. "Train" for training metric)
            ['main_subset_name']: Name of each scripts subset (-> e.g. "Train")
                .groups: Dictionary of subsets belonging to one of the scripts subsets, e.g. ["Recall", "NMI", ...]
                    ['specific_metric_name']: Specific name of the metric of interest, e.g. Recall@1.
        """
        if sub_loggers is None:
            sub_loggers = []
        self.prop = opt
        self.prefix = '{}_'.format(prefix) if prefix is not None else ''
        self.sub_loggers = sub_loggers

        # Make Logging Directories
        check_folder = opt.save_path
        if start_new:
            check_folder = init_logging(opt)

        # Set Graph and CSV writer
        self.csv_writer, self.graph_writer, self.progress_saver = {}, {}, {}
        for sub_logger in sub_loggers:
            csv_savepath = check_folder + '/CSVLogs'
            if not os.path.exists(csv_savepath):
                os.makedirs(csv_savepath)
            self.csv_writer[sub_logger] = CSVWriter(csv_savepath + '/Data_{}{}'.format(self.prefix, sub_logger))

            prgs_savepath = check_folder + '/ProgressionPlots'
            if not os.path.exists(prgs_savepath):
                os.makedirs(prgs_savepath)
            self.graph_writer[sub_logger] = InfoPlotter(prgs_savepath + '/Graph_{}{}'.format(self.prefix, sub_logger))
            self.progress_saver[sub_logger] = ProgressSaver()

        # WandB Init
        self.save_path = check_folder
        self.wandb_run = None

        try:
            # Auto-generate project and run names
            project_name = f"DAS_{opt.dataset.upper()}"
            run_name = f"{opt.arch}_{opt.loss}_{opt.batch_mining}"
            if opt.savename and opt.savename != 'group_plus_seed':
                run_name = f"{run_name}_{opt.savename}"
                
            # Initialize wandb
            self.wandb_run = wandb.init(
                project=project_name,
                name=run_name,
                config={
                    'dataset': opt.dataset,
                    'architecture': opt.arch,
                    'loss': opt.loss,
                    'batch_mining': opt.batch_mining,
                    'embed_dim': opt.embed_dim,
                    'learning_rate': opt.lr,
                    'batch_size': opt.bs,
                    'n_epochs': opt.n_epochs,
                    'optimizer': opt.optim,
                    'scheduler': opt.scheduler,
                    'seed': opt.seed,
                    'evaluation_metrics': opt.evaluation_metrics,
                    'storage_metrics': opt.storage_metrics,
                },
                tags=[opt.dataset, opt.arch, opt.loss, opt.batch_mining],
                dir=check_folder,
                reinit=True
            )
            print(f"WandB initialized: {self.wandb_run.name}")
        except Exception as e:
            print(f"Failed to initialize wandb: {e}")
            self.wandb_run = None

        # Tensorboard Init
        self.tensorboard = SummaryWriter(log_dir=check_folder)

    def log_to_wandb(self, metrics, step=None, prefix=""):
        """Log metrics to wandb"""
        if self.wandb_run is not None:
            try:
                # Add prefix to metric names if provided
                if prefix:
                    metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
                
                # Convert numpy types to Python types for wandb
                clean_metrics = {}
                for k, v in metrics.items():
                    if isinstance(v, np.integer):
                        clean_metrics[k] = int(v)
                    elif isinstance(v, np.floating):
                        clean_metrics[k] = float(v)
                    elif isinstance(v, np.ndarray):
                        clean_metrics[k] = v.tolist()
                    else:
                        clean_metrics[k] = v
                
                wandb.log(clean_metrics, step=step)
            except Exception as e:
                print(f"Failed to log to wandb: {e}")

    def log_best_metrics(self, sub_logger, step=None):
        """Log best metrics for a specific sub_logger"""
        if self.wandb_run is not None:
            try:
                best_metrics = {}
                for group_name, group_data in self.progress_saver[sub_logger].groups.items():
                    for metric_name, metric_data in group_data.items():
                        if metric_data['content']:
                            # Get the best value (max for most metrics, min for loss)
                            values = metric_data['content']
                            if 'loss' in metric_name.lower():
                                best_value = min(values)
                            else:
                                best_value = max(values)
                            best_metrics[f"best_{metric_name}"] = best_value
                
                if best_metrics:
                    self.log_to_wandb(best_metrics, step=step, prefix=f"best_{sub_logger}")
            except Exception as e:
                print(f"Failed to log best metrics to wandb: {e}")

    def update(self, *sub_loggers, update_all=False):
        online_content = []

        if update_all:
            sub_loggers = self.sub_loggers

        for sub_logger in list(sub_loggers):
            for group in self.progress_saver[sub_logger].groups.keys():
                pgs = self.progress_saver[sub_logger].groups[group]
                segments = pgs.keys()
                per_seg_saved_idxs = [pgs[segment]['saved_idx'] for segment in segments]
                per_seg_contents = [pgs[segment]['content'][idx:] for segment, idx in zip(segments, per_seg_saved_idxs)]
                per_seg_contents_all = [pgs[segment]['content'] for segment, idx in zip(segments, per_seg_saved_idxs)]

                # Adjust indexes
                for content, segment in zip(per_seg_contents, segments):
                    self.progress_saver[sub_logger].groups[group][segment]['saved_idx'] += len(content)

                tupled_seg_content = [list(seg_content_slice) for seg_content_slice in zip(*per_seg_contents)]

                self.csv_writer[sub_logger].log(group, segments, tupled_seg_content)
                self.graph_writer[sub_logger].make_plot(sub_logger, group, segments, per_seg_contents_all)

                for i, segment in enumerate(segments):
                    if group == segment:
                        name = sub_logger + ': ' + group
                    else:
                        name = sub_logger + ': ' + group + ': ' + segment
                    online_content.append((name, per_seg_contents[i]))

    def __del__(self):
        if self.tensorboard:
            self.tensorboard.close()
        if self.wandb_run is not None:
            try:
                wandb.finish()
            except:
                pass
