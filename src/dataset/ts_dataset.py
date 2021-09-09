import os
import torch
import pandas as pd
import torchvision as tv

FRAMERATE = 1  # default value


class TemporalShapeDataset(torch.utils.data.Dataset):

    def __init__(self, root, clip_size, is_val, transform_post=None,
                is_test=False, seq_first=False):
        self.root = root
        self.transform_post = transform_post

        self.clip_size = clip_size
        self.is_val = is_val
        self.seq_first = seq_first
        self.labels = pd.read_csv(os.path.join(self.root, 'labels.csv'))

    def __getitem__(self, index):
        """
        [!] FPS jittering doesn't work with AV dataloader as of now
        """

        ground_truth_bb = self.labels.loc[index][-4:].values
        ground_truth_bb = torch.from_numpy(ground_truth_bb).float()

        imgs = [tv.io.read_image(os.path.join(self.root, str(i)) + '.jpg').float()
                for i in range(self.clip_size)]

        transform_norm = tv.transforms.Compose([
            tv.transforms.Normalize([0.406], [0.225])
        ])

        imgs = [transform_norm(img) for img in imgs]

        # format data to torch
        data = torch.stack(imgs).float()
        if not self.seq_first:
            data = data.permute(1, 0, 2, 3)
        return data, ground_truth_bb

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    upscale_size = int(84 * 1.1)
    data_path = '../../data/test_100seqs_30_per_seq/'

    loader = TemporalShapeDataset(root=data_path,
                                  clip_size=30,
                                  is_val=False,
                                  )
    import time
    from tqdm import tqdm

    batch_loader = torch.utils.data.DataLoader(
        loader,
        batch_size=10, shuffle=False,
        num_workers=2, pin_memory=True)

    start = time.time()
    for i, a in enumerate(tqdm(batch_loader)):
        if i > 100:
            break
        pass
    print("Size --> {}".format(a[0].size()))
    print(time.time() - start)
