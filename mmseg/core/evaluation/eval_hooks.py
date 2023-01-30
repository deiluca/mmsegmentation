# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings

import torch.distributed as dist
from mmcv.runner import DistEvalHook as _DistEvalHook
from mmcv.runner import EvalHook as _EvalHook
from torch.nn.modules.batchnorm import _BatchNorm
# from mmseg.core.evaluation import pre_eval_to_metrics
import numpy as np

class EvalHook(_EvalHook):
    """Single GPU EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
        pre_eval (bool): Whether to use progressive mode to evaluate model.
            Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self,
                 *args,
                 by_epoch=False,
                 efficient_test=False,
                 pre_eval=False,
                 outdir=None,
                 **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.pre_eval = pre_eval
        self.latest_results = None
        self.outdir = outdir
        if efficient_test:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` for evaluation hook '
                'is deprecated, the evaluation hook is CPU memory friendly '
                'with ``pre_eval=True`` as argument for ``single_gpu_test()`` '
                'function')

    # def _do_evaluate_old(self, runner):
    #     """perform evaluation and save ckpt."""
    #     if not self._should_evaluate(runner):
    #         return

    #     from mmseg.apis import single_gpu_test
    #     # results = single_gpu_test(
    #     #     runner.model, self.dataloader, show=False, pre_eval=self.pre_eval, format_only=True, out_dir=None, format_args={"imgfile_prefix":self.outdir+"/x"})
    #     results = single_gpu_test(
    #         runner.model, self.dataloader, show=False, pre_eval=False, format_only=True, out_dir=None, format_args={"imgfile_prefix":self.outdir+"/x"})
    #     # self.dataloader.dataset.format_results(results, imgfile_prefix=)

    #     self.latest_results = results
    #     runner.log_buffer.clear()
    #     runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
    #     # key_score = self.evaluate(runner, results)
    #     # if self.save_best:
    #     #     self._save_ckpt(runner, key_score)


    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from mmseg.apis import single_gpu_test
        results, results_for_saving_imgs = single_gpu_test(
            runner.model, self.dataloader, show=False, pre_eval=self.pre_eval)

        # calculate metrics from pre_eval results
        metrics = [self.dataloader.dataset.return_pre_eval_metrics([r], ["mDice"])["Dice"][1] for r in results]
        print('Validation Dice scores:\n', '\n'.join([str(m) for m in metrics]))

        self.latest_results = results
        runner.log_buffer.clear()
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results) # here is the real evaluation taking place!!!

        # new by Luca: save images to disk at highest val score
        if self.save_best:
            best_score = runner.meta['hook_msgs'].get(
                'best_score', self.init_value_map[self.rule])
            if self.compare_func(key_score, best_score):
                best_score = key_score
                runner.meta['hook_msgs']['best_score'] = best_score
                self.dataloader.dataset.format_results(results_for_saving_imgs, file_info=metrics, imgfile_prefix=self.outdir)
                # self._save_ckpt(runner, key_score)
                # save empty file with mean dice score for class 1
                mean_dice_class_1 = sum(metrics)/len(metrics)
                file = self.outdir+f'/mean_dice_score_cls1_at_lowest_mean_dice_allclasses_{mean_dice_class_1:.2f}.txt'
                open(file, 'a').close()

        print("...")


class DistEvalHook(_DistEvalHook):
    """Distributed EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
        pre_eval (bool): Whether to use progressive mode to evaluate model.
            Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self,
                 *args,
                 by_epoch=False,
                 efficient_test=False,
                 pre_eval=False,
                 **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.pre_eval = pre_eval
        self.latest_results = None
        if efficient_test:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` for evaluation hook '
                'is deprecated, the evaluation hook is CPU memory friendly '
                'with ``pre_eval=True`` as argument for ``multi_gpu_test()`` '
                'function')

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from mmseg.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect,
            pre_eval=self.pre_eval)
        self.latest_results = results
        runner.log_buffer.clear()

        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)
