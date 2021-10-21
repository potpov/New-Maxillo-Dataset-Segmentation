import torch
from torch import nn
from tqdm import tqdm
from torch.nn.functional import interpolate
from augmentations import CenterCrop
import numpy as np
import torchio as tio
import logging
from utils import resample
import cc3d


def test(model, test_loader, epoch, writer, evaluator, phase):

    model.eval()

    with torch.no_grad():
        evaluator.reset_eval()
        for i, (subject, loader) in tqdm(enumerate(test_loader), total=len(test_loader), desc='val epoch {}'.format(str(epoch))):
            aggr = tio.inference.GridAggregator(subject, overlap_mode='average')
            for subvolume in loader:
                # batchsize with torchio affects the number of grids we extract from a patient.
                # when we aggragate the patient the volume is just one.
                images = subvolume['data'][tio.DATA].float().cuda()  # BS, 3, Z, H, W

                output = model(images)  # BS, Classes, Z, H, W

                aggr.add_batch(output, subvolume[tio.LOCATION])

            output = aggr.get_output_tensor()  # C, Z, H, W
            labels = subject[0]['label']  # original labels from storage


            output = interpolate(output.unsqueeze(0), size=tuple(labels.shape), mode='trilinear', align_corners=False)
            origi_vol = interpolate(subject.subject['data'][tio.DATA].unsqueeze(0), size=tuple(labels.shape), mode='trilinear', align_corners=False)
            origi_vol = origi_vol.squeeze().cpu().detach().numpy()
            output = output.squeeze().cpu().detach().numpy()

            assert output.shape == labels.shape, f"{output.shape} != {labels.shape}"

            # final predictions
            if output.ndim > 3:
                output = torch.argmax(torch.nn.Softmax(dim=0)(output), dim=0).numpy()
            else:
                output = nn.Sigmoid()(torch.from_numpy(output))  # BS, 1, Z, H, W
                output = torch.where(output > 0.5, 1, 0)
                output = output.squeeze().cpu().detach().numpy()  # BS, Z, H, W

            # post-processing for noise
            if phase in ['Test', 'Final']:
                output = cc3d.connected_components(output)
                two_biggest = np.argsort([np.sum(output == l) for l in np.unique(output)])[-3:-1]
                output = np.where(np.logical_and(output != two_biggest[0], output != two_biggest[1]), 0, 1)

            evaluator.compute_metrics(output, labels, origi_vol, str(i), phase)

            # TB DUMP FOR BINARY CASE!
            # images = np.clip(images, 0, None)
            # images = (images.asphase(np.float))/images.max()
            # if writer is not None:
            #     unempty_idx = np.argwhere(np.sum(labels != config['labels']['BACKGROUND'], axis=(0, 2)) > 0)
            #     randidx = np.random.randint(0, unempty_idx.size - 1, 5)
            #     rand_unempty_idx = unempty_idx[randidx].squeeze()  # random slices from unempty ones
            #
            #     dump_img = np.concatenate(np.moveaxis(images[:, rand_unempty_idx], 0, 1))
            #
            #     dump_gt = np.concatenate(np.moveaxis(labels[:, rand_unempty_idx], 0, 1))
            #     dump_pred = np.concatenate(np.moveaxis(output[:, rand_unempty_idx], 0, 1))
            #
            #     dump_img = np.stack((dump_img, dump_img, dump_img), axis=-1)
            #     a = dump_img.copy()
            #     a[dump_pred == config['labels']['INSIDE']] = (0, 0, 1)
            #     b = dump_img.copy()
            #     b[dump_gt == config['labels']['INSIDE']] = (0, 0, 1)
            #     dump_img = np.concatenate((a, b), axis=-2)
            #     writer.add_image(
            #         "3D_results",
            #         dump_img,
            #         len(test_loader) * epoch + i,
            #         dataformats='HWC'
            #     )
            # END OF THE DUMP

    epoch_iou, epoch_dice, epoch_haus = evaluator.mean_metric(phase=phase)
    if writer is not None and phase != "Final":
        writer.add_scalar(f'{phase}/IoU', epoch_iou, epoch)
        writer.add_scalar(f'{phase}/Dice', epoch_dice, epoch)
        writer.add_scalar(f'{phase}/Hauss', epoch_haus, epoch)

    if phase in ['Test', 'Final']:
        logging.info(
            f'{phase} Epoch [{epoch}], '
            f'{phase} Mean Metric (IoU): {epoch_iou}'
            f'{phase} Mean Metric (Dice): {epoch_dice}'
            f'{phase} Mean Metric (haus): {epoch_haus}'
        )

    return epoch_iou, epoch_dice, epoch_haus
