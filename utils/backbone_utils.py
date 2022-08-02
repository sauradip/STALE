import torch
import numpy as np 
import yaml

with open("./config/anet.yaml", 'r', encoding='utf-8') as f:
        tmp = f.read()
        config = yaml.load(tmp, Loader=yaml.FullLoader)

device = torch.device("cuda")

# def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)


def interpolate_pos_embed(model, checkpoint_model):
    # video
    num_frames = 16
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]  # channel dim
        num_patches = model.patch_embed.num_patches  #
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches  # 0/1

        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int((num_patches // (num_frames // model.patch_embed.tubelet_size)) ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            # B, L, C -> BT, H, W, C -> BT, C, H, W
            pos_tokens = pos_tokens.reshape(-1, num_frames // model.patch_embed.tubelet_size, orig_size, orig_size,
                                            embedding_size)
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, num_frames // model.patch_embed.tubelet_size,
                                                                new_size, new_size, embedding_size)
            pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed



def prepare_backbone(model,checkpoint):

        if 'model' in checkpoint:
            raw_checkpoint_model = checkpoint['model']
        elif 'module' in checkpoint:
            raw_checkpoint_model = checkpoint['module']
        else:
            raw_checkpoint_model = checkpoint


        if config['pretraining']['isPretrain'] == 1:
            checkpoint_model = OrderedDict()
            for k, v in raw_checkpoint_model.items():
                if k.startswith('encoder.'):
                    checkpoint_model[k[8:]] = v  # remove 'encoder.' prefix
            del checkpoint_model['norm.weight']
            del checkpoint_model['norm.bias']
        elif config['pretraining']['isPretrain'] == 0:
            checkpoint_model = raw_checkpoint_model
        else:
            raise ValueError("Warning: Double Check!")


        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        interpolate_pos_embed(model, checkpoint_model)

        msg = model.load_state_dict(checkpoint_model, strict=False)
        # print(msg)

        for name, p in model.named_parameters():
            if name in msg.missing_keys:
                p.requires_grad = True
            else:
                p.requires_grad = False if not args.fulltune else True
        # for _, p in model.head.named_parameters():
        #     p.requires_grad = True
        # for _, p in model.fc_norm.named_parameters():
        #     p.requires_grad = True

        model.to(device)

        return model
