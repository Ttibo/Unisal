

class UCFDataset(DHF1KDataset):

    source = 'UCFSports'
    dynamic = True

    img_channels = 1
    n_train_val_videos = 103
    test_vid_nrs = (1, 47)
    frame_rate = 24

    def __init__(self, out_size=(256, 384), val_size=10, n_images_file=None,
                 seq_per_vid_val=1, register_file='ucfsports_register.json',
                 phase='train',
                 frame_modulo=4,
                 seq_len=12,
                 **kwargs):
        self.phase_str = 'test' if phase in ('eval', 'test') else 'train'
        self.register_file = self.phase_str + "_" + register_file
        self.register = None
        super().__init__(out_size=out_size, val_size=val_size,
                         n_images_file=n_images_file,
                         seq_per_vid_val=seq_per_vid_val,
                         x_val_seed=27, target_size=out_size,
                         frame_modulo=frame_modulo, phase=phase,
                         seq_len=seq_len,
                         **kwargs)
        if phase in ('eval', 'test'):
            self.target_size_dict = self.get_register()['vid_size_dict']

    @property
    def n_images_dict(self):
        if self._n_images_dict is None:
            self._n_images_dict = self.get_register()['n_images_dict']
            self._n_images_dict = {vid_nr: ni for vid_nr, ni
                                   in self._n_images_dict.items()
                                   if vid_nr in self.vid_nr_array}
        return self._n_images_dict

    def get_register(self):
        if self.register is None:
            register_file = config_path / self.register_file
            if register_file.exists():
                with open(config_path / register_file, 'r') as f:
                    self.register = json.load(f)
                for reg_key in ('n_images_dict', 'vid_name_dict',
                                'vid_size_dict'):
                    self.register[reg_key] = {
                        int(key): val for key, val in
                        self.register[reg_key].items()}
            else:
                self.register = self.generate_register()
                with open(config_path / register_file, 'w') as f:
                    json.dump(self.register, f, indent=2)
        return self.register

    def generate_register(self):
        n_images_dict = {}
        vid_name_dict = {}
        vid_size_dict = {}

        for vid_idx, folder in enumerate(sorted(self.dir.glob('*-*'))):
            vid_nr = vid_idx + 1
            vid_name_dict[vid_nr] = folder.stem
            image_files = list((folder / 'images').glob('*.png'))
            n_images_dict[vid_nr] = len(image_files)
            img = cv2.imread(str(image_files[0]))
            vid_size_dict[vid_nr] = tuple(img.shape[:2])

        return dict(
            vid_name_dict=vid_name_dict, n_images_dict=n_images_dict,
            vid_size_dict=vid_size_dict)

    def preprocess_sequence(self, frame_seq, dkey, vid_nr):
        transformations = [
            transforms.ToPILImage()
        ]

        vid_size = self.register['vid_size_dict'][vid_nr]
        interpolation = PIL.Image.LANCZOS if dkey in ('frame', 'sal')\
            else PIL.Image.NEAREST
        out_size_ratio = self.out_size[1] / self.out_size[0]
        this_size_ratio = vid_size[1] / vid_size[0]
        if this_size_ratio < out_size_ratio:
            size = (int(self.out_size[1] / this_size_ratio), self.out_size[1])
        else:
            size = (self.out_size[0], int(self.out_size[0] * this_size_ratio))
        transformations.append(
            transforms.Resize(size, interpolation=interpolation))

        transformations += [
            transforms.CenterCrop(self.out_size),
            transforms.ToTensor(),
        ]

        if dkey == 'frame' and 'rgb_mean' in self.preproc_cfg:
            transformations.append(
                transforms.Normalize(
                    self.preproc_cfg['rgb_mean'], self.preproc_cfg['rgb_std']))
        elif dkey == 'sal':
            transformations.append(transforms.Lambda(utils.normalize_tensor))
        elif dkey == 'fix':
            transformations.append(
                transforms.Lambda(lambda fix: torch.gt(fix, 0.5)))

        processing = transforms.Compose(transformations)

        tensor = [processing(img) for img in frame_seq]
        tensor = torch.stack(tensor)
        return tensor

    preprocess_sequence_eval = HollywoodDataset.preprocess_sequence_eval

    def get_annotation_dir(self, vid_nr):
        vid_name = self.register['vid_name_dict'][vid_nr]
        return self.dir / vid_name

    def get_data_file(self, vid_nr, f_nr, dkey):
        if dkey == 'frame':
            folder = 'images'
        elif dkey == 'sal':
            folder = 'maps'
        elif dkey == 'fix':
            folder = 'fixation'
        else:
            raise ValueError(f'Unknown data key {dkey}')
        vid_name = self.register['vid_name_dict'][vid_nr]
        return self.get_annotation_dir(vid_nr) / folder /\
            f"{vid_name[:-4]}_{vid_name[-3:]}_{f_nr:03d}.png"

    get_seq = HollywoodDataset.get_seq

    @property
    def dir(self):
        if self._dir is None:
            self._dir = Path(os.environ["UCFSPORTS_DATA_DIR"]) /\
                        ('training' if self.phase in ('train', 'valid')
                         else 'testing')
        return self._dir


def get_optimal_out_size(img_size):
    ar = img_size[0] / img_size[1]
    min_prod = 100
    max_prod = 120
    ar_array = []
    size_array = []
    for n1 in range(7, 14):
        for n2 in range(7, 14):
            if min_prod <= n1 * n2 <= max_prod:
                this_ar = n1 / n2
                this_ar_ratio = min((ar, this_ar)) / max((ar, this_ar))
                ar_array.append(this_ar_ratio)
                size_array.append((n1, n2))

    max_ar_ratio_idx = np.argmax(np.array(ar_array)).item()
    bn_size = size_array[max_ar_ratio_idx]
    out_size = tuple(r * 32 for r in bn_size)
    return out_size
