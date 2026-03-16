import os
import sys
import tqdm
import sys; sys.path.insert(0, '.')
import argparse
import numpy as np
import transformers
from sacred import Experiment

import torch
import model.metric as module_metric
import data_loader.data_loader as module_data
from utils import state_dict_data_parallel_fix
from parse_config import ConfigParser

import time
import decord

ex = Experiment('test')


def feature_path(root, clip_uid):
    return os.path.join(root, clip_uid + '.npy')


def save_feature_array(path, array):
    np.save(path, array)

@ex.main
def run():
    # setup data_loader instances
    config._config['data_loader']['type'] = 'TextVideoDataLoader'
    config._config['data_loader']['args']['split'] = args.split
    config._config['data_loader']['args']['tsfm_split'] = 'test'  # set transform to test split to remove augmentations
    config._config['data_loader']['args']['shuffle'] = False
    config._config['data_loader']['args']['batch_size'] = args.batch_size
    config._config['data_loader']['args']['sliding_window_stride'] = args.sliding_window_stride
    config._config['data_loader']['args']['subsample'] = args.subsample

    data_loader = config.initialize('data_loader', module_data)

    tokenizer = transformers.AutoTokenizer.from_pretrained(config['arch']['args']['text_params']['model'])

    # build model architecture
    import model.model as module_arch
    model = config.initialize('arch', module_arch)

    # get function handles of loss and metrics
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # logger.info('Loading checkpoint: {} ...'.format(config.resume))

    if os.path.exists(config.resume):
        print('load from', config.resume)
        checkpoint = torch.load(config.resume)
        state_dict = checkpoint['state_dict']
        new_state_dict = state_dict_data_parallel_fix(state_dict, model.state_dict())
        model.load_state_dict(new_state_dict, strict=True)
    else:
        print('Using random weights')

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    print(len(data_loader))

    if not os.path.exists(args.save_feats):
        os.makedirs(args.save_feats)

    # extract clip features
    if args.subsample == 'video' and args.d == 0:
        num_frame = config.config['data_loader']['args']['video_params']['num_frames']
        dim = config.config['arch']['args']['projection_dim']
        with torch.no_grad():
            for i, data in enumerate(tqdm.tqdm(data_loader)):
                # leave this for now since not doing anything on the gpu
                # pdb.set_trace()
                out_path = feature_path(args.save_feats, data['meta']['clip_uid'][0])
                if os.path.exists(out_path):
                    print(f"{data['meta']['clip_uid']} is already.")
                    continue
                # this implementation is cautious, we use 4f video-encoder to extract featurs of whole clip.
                f, c, h, w = data['video'].shape[1], data['video'].shape[2], data['video'].shape[3], data['video'].shape[4]

                data['video'] = data['video'][0][:(f // num_frame * num_frame), ]
                data['video'] = data['video'].reshape(-1, num_frame, c, h, w)

                print(data['video'].shape)
                # data['video'] = data['video'] #.to(device)
                outs = torch.zeros(data['video'].shape[0], dim)

                batch = 16
                if data['video'].shape[0] % batch == 0:
                    times = data['video'].shape[0] // batch
                else:
                    times = data['video'].shape[0] // batch + 1

                for j in range(times):
                    start = j*batch
                    if (j+1) * batch > data['video'].shape[0]:
                        end = data['video'].shape[0]
                    else:
                        end = (j+1)*batch

                    outs[start:end,] = \
                        model.compute_video(data['video'][start:end,].to(device)).cpu().detach()

                outs = outs.detach().cpu().numpy()
                save_feature_array(
                    out_path,
                    outs,
                )
                print(f"Saved {data['meta']['clip_uid']}. ")

    if args.subsample == 'video' and args.d != 0:
        num_frame = config.config['data_loader']['args']['video_params']['num_frames']
        dim = config.config['arch']['args']['projection_dim']
        dataset = data_loader.dataset
        if args.data is not None:
            dataset.data_dir = args.data
        with torch.no_grad():
            for i in tqdm.trange(len(dataset.metadata)):
                sample = dataset.metadata.iloc[i]
                meta_arr = {'video_uid': sample[0], 'clip_uid': sample[1]}

                out_path = feature_path(args.save_feats, meta_arr['clip_uid'])
                if os.path.exists(out_path):
                    print(f"{meta_arr['clip_uid']} is already.")
                    continue

                video_fp, rel_fp = dataset._get_video_path(sample)
                video_reader = decord.VideoReader(video_fp, num_threads=1)
                # frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
                video_reader.skip_frames(1)

                start = sample[2] * 30
                end = sample[3] * 30
                max_frame = len(video_reader)
                end = min(end, max_frame)
                nframes = int(end - start)
                if sample[0] == sample[1]:
                    start = 0

                idx = torch.arange(0, nframes, 8) # * 32
                idx = idx[:, None] + torch.arange(32)[::args.d]
                loc = ~(idx >= nframes).any(-1)
                idx = idx[loc]
                print(nframes, idx.shape)

                outs = torch.zeros(idx.shape[0], dim)

                batch_size = 32
                nbatch = (idx.size(0) + batch_size - 1) // batch_size
                for bi in tqdm.trange(nbatch):
                    subset = idx[bi*batch_size : (bi+1)*batch_size].detach().numpy()
                    tlist = start + np.unique(subset)  #.detach().numpy()
                    remap = { tlist[i]: i for i in range(len(tlist)) }
                    subset_remapped = np.array([remap[x] for x in subset.reshape(-1)])
                    subset_remapped = subset_remapped.reshape(subset.shape)
                    subset_remapped = torch.from_numpy(subset_remapped)

                    frames = video_reader.get_batch(tlist) #.detach().numpy())
                    frames = frames.float() / 255
                    frames = frames.permute(0, 3, 1, 2)

                    if dataset.transforms is not None:
                        frames = frames.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
                        frames = dataset.transforms(frames)
                        frames = frames.transpose(0, 1)  # recover

                    frame_window = frames[subset_remapped].to(device)
                    # frames = frames.to(device)
                    outs[bi*batch_size: (bi+1)*batch_size] = \
                        model.compute_video(frame_window).cpu().detach()

                    # if bi == 10:
                    #     break

                save_feature_array(
                    out_path,
                    outs.cpu().numpy(),
                )
                print(f"Saved {meta_arr['clip_uid']}. ")

    # extract text features
    if args.subsample == 'text':
        cache_bert_feature = dict()
        cache_bert_feature_768 = dict()
        with torch.no_grad():
            for i, data in enumerate(tqdm.tqdm(data_loader)):
                raw_text = data['text'][0]
                if tokenizer is not None:
                    data['text'] = tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                data['text'] = {key: val.to(device) for key, val in data['text'].items()}

                if args.token:
                    # text_embed = model.compute_text_tokens(data['text'])[0]
                    # num_words = data['text']['attention_mask'][0].sum()
                    # text_embed = text_embed[1 : num_words-1]
                    text_embed, text_emb_768 = model.compute_text_tokens(data['text'])
                    num_words = data['text']['attention_mask'][0].sum()
                    text_embed = text_embed[0, 1 : num_words-1]
                    text_emb_768 = text_emb_768[0, 1 : num_words-1]
                else:
                    # raise NotImplementedError()
                    text_embed, text_emb_768 = model.compute_text(data['text'])

                if raw_text not in cache_bert_feature.keys():
                    cache_bert_feature[raw_text] = text_embed.cpu().detach()
                    cache_bert_feature_768[raw_text] = text_emb_768.cpu().detach()

        np.save(os.path.join(args.save_feats, f'sentence_256_{args.split}.npy'), cache_bert_feature)
        np.save(os.path.join(args.save_feats, f'sentence_768_{args.split}.npy'), cache_bert_feature_768)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume',
                      default='pretrained/egovlp.pth',
                      help='path to latest checkpoint (default: None)')
    args.add_argument('--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('--sliding_window_stride', default=-1, type=int,
                      help='test time temporal augmentation, repeat samples with different start times.')
    args.add_argument('-subsample', '--subsample', default='video', type=str, # 0 for vidoe while 1 for text.
                      help='source data from video or text.')
    args.add_argument('--token', default=False, type=bool,
                      help='whether use token features to represent sentence.')
    args.add_argument('--save_feats', default='dataset/ego4d/benchmark_splits/nlq/egovlp',
                      help='path to store text & video feats, this is for saving embeddings if you want to do offline retrieval.')
    args.add_argument('--split', default='val', choices=['train', 'val', 'test'],
                      help='split to evaluate on.')
    args.add_argument('--batch_size', default=1, type=int,
                      help='size of batch')
    args.add_argument('-gpu', '--gpu', default=0, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--data', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--metadir', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--s', default=None, type=int,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--l', default=None, type=int,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--d', default=0, type=int,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--task', default='nlq', type=str,
                      help='indices of GPUs to enable (default: all)')
    # args.add_argument('--fast', action='store_true')
    config = ConfigParser(args, test=True, eval_mode='nlq')
    # hack to get sliding into config
    args = args.parse_args()
    config._config['sliding_window_stride'] = args.sliding_window_stride
    ex.add_config(config.config)


    os.environ["CUDA_VISIBLE_DEVICES"] =  ""+str(args.gpu)
    if args.data is not None:
        os.environ['EGO4D_PATH'] = str(args.data)
    if args.metadir is not None:
        os.environ['METADATA_DIR'] = str(args.metadir)
    if args.s is not None:
        os.environ['METADATA_START'] = str(args.s)
    if args.l is not None:
        os.environ['METADATA_LEN'] = str(args.l)
    if args.d is not None:
        os.environ['STRIDE'] = str(args.d)
    os.environ['SAVEPATH'] = str(args.save_feats)
    os.environ['TASK'] = args.task
    # os.environ['FAST'] = str(args.fast)

    run()
    # ex.run()
