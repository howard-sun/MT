# *_*coding:utf-8 *_*
import os
import argparse
import time

import torch

from . import data_set
from . import models
from . import optim
from . import utils


class Trainer(object):
    def __init__(self, args, data_splits, device):

        self.args = args
        self.data_splits = data_splits
        self.device = device
        self.cpu_only = (device == torch.device('cpu'))

        self.model = models.transformer.Transformer(
            args=args,
            src_dict=data_splits.vocab_src,
            tgt_dict=data_splits.vocab_tgt,
        ).to(device)

        if args.optim == 'adam':
            self.optimizer = torch.optim.Adam(
                params=self.model.parameters(),
                lr=args.learning_rate,
                betas=args.betas,
                weight_decay=args.weight_decay
            )
        elif args.optim == 'adamw':
            self.optimizer = torch.optim.AdamW(
                params=self.model.parameters(),
                lr=args.learning_rate,
                betas=args.betas,
                weight_decay=args.weight_decay
            )
        else:
            raise ValueError

        if args.fp16:
            raise NotImplementedError('fp16 is disallowed!')

        self.scheduler = None
        if args.decay_method == 'inverse_sqrt' and not args.fp16:
            self.scheduler = optim.lr_scheduler.InverseSqrtScheduler(
                self.optimizer,
                warmup_steps=args.warmup_steps,
                min_lr=args.min_lr
            )


        self.test_loader = data_set.get_dataloader(
            dset=data_splits['tst'],
            batch_size=args.eval_batch_size,
            shuffle=False,
            pin_memory=not self.cpu_only
        )

    def test(self, path):
        self.model.eval()
        out_dir = os.path.dirname(path)
        os.makedirs(out_dir, exist_ok=True)
        vocab_tgt = self.data_splits.vocab_tgt
        with open(path, 'w') as outfile:
            begin_time = time.time()
            for idx, batch in enumerate(
                    utils.yield_to_device(self.test_loader, self.device)):
                src, src_lens, tgt_in, tgt_out, tgt_lens = batch
                src_len = src_lens.max().cpu().item()
                tgt_len = int(self.args.max_decode_length_multiplier *
                              src_len + self.args.max_decode_length_base)
                with torch.no_grad():
                    if self.args.decode_method == 'greedy':
                        decoded = self.model.greedy_decode(src,
                                                           max_length=tgt_len)
                    elif self.args.decode_method == 'beam':
                        decoded = self.model.beam_decode(
                            src,
                            beam_size=self.args.beam_size,
                            length_normalize=self.args.length_normalize,
                            max_length=tgt_len)
                    else:
                        raise NotImplementedError()
                    decoded = decoded.cpu().numpy()
                    for seq in decoded:
                        tks = vocab_tgt.decode_ids(seq, dettach_ends=True)
                        outfile.write('{}\n'.format(' '.join(tks)))
                cur_time = time.time()
                print(('\r[step {:4d}/{:4d}] time: {:.1f}s').format(
                    idx + 1,
                    len(self.test_loader),
                    cur_time - begin_time),
                      end='')

    def save(self, path, epoch=None, verbose=True):
        os.makedirs(path, exist_ok=True)
        if epoch is not None:
            save_path = os.path.join(path, 'model{}.pth'.format(epoch))
        else:
            save_path = os.path.join(path, 'model.pth')

        # Convert model to fp32 if using mixed precision
        if self.args.fp16:
            self.model.float()
        torch.save(self.model.state_dict(), save_path)

        if self.args.fp16:
            self.model.half()

        if verbose:
            print('[*] Model is saved in \'{}\'.'.format(save_path))

    def load(self, path):
        if self.args.fp16:
            self.model.float()
        self.model.load_state_dict(torch.load(path, map_location=self.device))

        if self.args.fp16:
            self.model.half()

        print('[*] Model is loaded from \'{}\''.format(path))


def main():
    args = parse_args()
    data_splits = data_set.SplittedDataset(args.data_dir,lang_src=args.lang_src,lang_tgt=args.lang_tgt,lowercase=args.lowercase)

    cuda_device = 'cuda:{}'.format(args.gpu)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(cuda_device if use_cuda else 'cpu')

    trainer = Trainer(args, data_splits, device)
    if args.init_checkpoint:
        trainer.load(args.init_checkpoint[0])
    if args.mode == 'test':
        assert args.output_path
        assert args.init_checkpoint
        trainer.test(args.output_path)
    else:
        raise NotImplementedError()

def parse_args():
    parser = argparse.ArgumentParser()
    # 环境参数
    parser.add_argument('--mode', choices=('train', 'test', 'val'), default='train')
    parser.add_argument('--fp16', action='store_true', help='Use fp16')
    parser.add_argument('--gpu', type=int, default=0)

    # 数据参数
    parser.add_argument('--lowercase', action='store_true')
    parser.add_argument('--data-dir', default='data/iwslt-2014')
    parser.add_argument('--lang-src', default='en')
    parser.add_argument('--lang-tgt', default='de')

    # 优化参数
    parser.add_argument('--max-epochs', type=int, default=30)
    parser.add_argument('--learning-rate', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.997))
    parser.add_argument('--optim', choices=('adam', 'adamw'), default='adamw')
    parser.add_argument('--decay-method', choices=('inverse_sqrt', 'cos'), default='inverse_sqrt')
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--min-lr', type=float, default=1e-9)
    parser.add_argument('--warmup-steps', type=int, default=8000)
    parser.add_argument('--batch-size', type=int, default=80)
    parser.add_argument('--eval-batch-size', type=int, default=20)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--label-smoothing', type=float, default=0.)
    parser.add_argument('--gradient-accumulation', type=int, default=2)

    # 训练参数
    parser.add_argument('--init-checkpoint', nargs='+')
    parser.add_argument('--output-path')
    parser.add_argument('--decode-method', default='greedy')
    parser.add_argument('--max-decode-length-multiplier', type=float, default=2)
    parser.add_argument('--max-decode-length-base', type=int, default=10)
    parser.add_argument('--length-normalize', type=bool, default=True)

    # 模型参数
    parser.add_argument('--dec-embed-dim', type=int, default=512)
    parser.add_argument('--dec-ffn-dim', type=int, default=1024)
    parser.add_argument('--dec-num-heads', type=int, default=4)
    parser.add_argument('--dec-num-layers', type=int, default=6)
    parser.add_argument('--dec-layernorm-before', action='store_true')
    parser.add_argument('--enc-embed-dim', type=int, default=512)
    parser.add_argument('--enc-ffn-dim', type=int, default=1024)
    parser.add_argument('--enc-num-heads', type=int, default=4)
    parser.add_argument('--enc-num-layers', type=int, default=6)
    parser.add_argument('--enc-layernorm-before', action='store_true')
    parser.add_argument('--dec-tied-weight', type=bool, default=True)

    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--act-dropout', type=float, default=0.1)
    parser.add_argument('--attn-dropout', type=float, default=0.0)
    parser.add_argument('--embed-dropout', type=float, default=0.3)

    parser.add_argument('--save-path', type=str, default='./save/')
    parser.add_argument('--save-epochs', type=int, default=1)

    return parser.parse_args()

    return parser.parse_args()




if __name__ == '__main__':
    main()
