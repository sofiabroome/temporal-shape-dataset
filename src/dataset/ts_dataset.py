import os
import torch
import pandas as pd
from PIL import Image
import torchvision as tv

FRAMERATE = 1  # default value


class TemporalShapeDataset(torch.utils.data.Dataset):

    def __init__(self, root, clip_size, is_val, nb_labels,
                 transform_post=None, is_test=False, seq_first=False):
        self.root = root
        self.transform_post = transform_post
        self.clip_size = clip_size
        self.is_val = is_val
        self.is_test = is_test
        self.seq_first = seq_first
        self.labels = pd.read_csv(os.path.join(self.root, 'labels.csv'))
        self.nb_labels = nb_labels

    def __getitem__(self, index):

        ground_truth = self.labels.loc[index][-self.nb_labels:].values
        ground_truth = torch.as_tensor(ground_truth, dtype=torch.float32)

        images = [Image.open(os.path.join(self.root, str(frame_ind)) + '.jpg')
                  for frame_ind in range(self.clip_size)]

        # transform_norm = tv.transforms.Compose([
        #     tv.transforms.ToTensor(),  # Scales to [0,1] and converts to tensor.
        #     tv.transforms.Normalize([0.5], [0.5])
        # ])
        transform_norm = tv.transforms.Compose([
            tv.transforms.ToTensor()  # Scales to [0,1] and converts to tensor.
        ])
        images = [transform_norm(img) for img in images]

        data = torch.stack(images)

        if not self.seq_first:
            data = data.permute(1, 0, 2, 3)
        return data, ground_truth

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
