from dataclasses import replace






def make_dit_builder(model, base_cfg: 'DiT2DConfig'):
    """
    :return: build() > (model, cfg)
    """




    def build(patch=None, img_size=None, in_channels=None, out_channels=None):
        cfg = base_cfg
        if patch is not None:
            cfg = replace(cfg, patch=patch)
        if img_size is not None:
            cfg = replace(cfg, img_size=img_size)
        if in_channels is not None:
            cfg = replace(cfg, in_channels=in_channels)
        if out_channels is not None:
            cfg = replace(cfg, out_channels=out_channels)

        m = model(cfg)

        return m, cfg

    return build