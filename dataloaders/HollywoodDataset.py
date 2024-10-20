


class HollywoodDataset(DHF1KDataset):

    source = 'Hollywood'
    dynamic = True

    img_channels = 1
    n_videos = {
        'train': 747,
        'test': 884
    }
    test_vid_nrs = (1, 884)
    frame_rate = 24

    def __init__(self, out_size=(224, 416), val_size=75, n_images_file=None,
                 seq_per_vid_val=1, register_file='hollywood_register.json',
                 phase='train',
                 frame_modulo=4,
                 seq_len=12,
                 **kwargs):
        self.register = None
        self.phase_str = 'test' if phase in ('eval', 'test') else 'train'
        self.register_file = self.phase_str + "_" + register_file
        super().__init__(out_size=out_size, val_size=val_size,
                         n_images_file=n_images_file,
                         seq_per_vid_val=seq_per_vid_val,
                         x_val_seed=42, phase=phase, target_size=out_size,
                         frame_modulo=frame_modulo,
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
                                   if vid_nr // 100 in self.vid_nr_array}
        return self._n_images_dict

    def get_register(self):
        if self.register is None:
            register_file = config_path / self.register_file
            if register_file.exists():
                with open(config_path / register_file, 'r') as f:
                    self.register = json.load(f)
                for reg_key in ('n_images_dict', 'start_image_dict',
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
        n_shots = {
            vid_nr: 0 for vid_nr in range(1, self.n_videos[self.phase_str] + 1)}
        n_images_dict = {}
        start_image_dict = {}
        vid_size_dict = {}

        for folder in sorted(self.dir.glob('actionclip*')):
            name = folder.stem
            vid_nr_start = 10 + len(self.phase_str)
            vid_nr = int(name[vid_nr_start:vid_nr_start + 5])
            shot_nr = int(name[-2:].replace("_", ""))
            n_shots[vid_nr] += 1

            vid_nr_shot_nr = 100 * vid_nr + shot_nr
            image_files = sorted((folder / 'images').glob('actionclip*.png'))
            n_images_dict[vid_nr_shot_nr] = len(image_files)
            start_image_dict[vid_nr_shot_nr] = int(image_files[0].stem[-5:])
            img = cv2.imread(str(image_files[0]))
            vid_size_dict[vid_nr_shot_nr] = tuple(img.shape[:2])

        return dict(
            n_shots=n_shots, n_images_dict=n_images_dict,
            start_image_dict=start_image_dict, vid_size_dict=vid_size_dict)

    def preprocess_sequence(self, frame_seq, dkey, vid_nr):
        transformations = [
            transforms.ToPILImage()
        ]

        vid_size = self.register['vid_size_dict'][vid_nr]
        if vid_size[0] != self.out_size[0]:
            interpolation = PIL.Image.LANCZOS if dkey in ('frame', 'sal')\
                else PIL.Image.NEAREST
            size = (self.out_size[0],
                    int(vid_size[1] * self.out_size[0] / vid_size[0]))
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

    def preprocess_sequence_eval(self, frame_seq, dkey, vid_nr):
        transformations = []

        if dkey == 'frame':
            transformations.append(transforms.ToPILImage())
            transformations.append(
                transforms.Resize(
                    self.out_size, interpolation=PIL.Image.LANCZOS))

        transformations.append(transforms.ToTensor())
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

    def get_annotation_dir(self, vid_nr_shot_nr):
        vid_nr = vid_nr_shot_nr // 100
        shot_nr = vid_nr_shot_nr % 100
        return self.dir / f"actionclip{self.phase_str}{vid_nr:05d}_{shot_nr:1d}"

    def get_data_file(self, vid_nr_shot_nr, f_nr, dkey):
        if dkey == 'frame':
            folder = 'images'
        elif dkey == 'sal':
            folder = 'maps'
        elif dkey == 'fix':
            folder = 'fixation'
        else:
            raise ValueError(f'Unknown data key {dkey}')
        vid_nr = vid_nr_shot_nr // 100
        f_nr += self.register['start_image_dict'][vid_nr_shot_nr] - 1
        return self.get_annotation_dir(vid_nr_shot_nr) / folder /\
            f'actionclip{self.phase_str}{vid_nr:05d}_{f_nr:05d}.png'

    def get_seq(self, vid_nr, frame_nrs, dkey):
        data_seq = [self.load_data(vid_nr, f_nr, dkey) for f_nr in frame_nrs]
        preproc_fun = self.preprocess_sequence if self.phase \
            in ('train', 'valid') else self.preprocess_sequence_eval
        return preproc_fun(data_seq, dkey, vid_nr)

    @property
    def dir(self):
        if self._dir is None:
            self._dir = Path(os.environ["HOLLYWOOD_DATA_DIR"]) /\
                        ('training' if self.phase in ('train', 'valid')
                         else 'testing')
        return self._dir
