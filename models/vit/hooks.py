import torch


def get_activation(name, activations):
    def hook(model, input, output):
        activations[name] = output

    return hook


def get_attention(name, attention):
    def hook(module, input, output):
        x = input[0]
        B, N, C = x.shape
        qkv = (
            module.qkv(x)
            .reshape(B, N, 3, module.num_heads, C // module.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * module.scale

        attn = attn.softmax(dim=-1)  # [:,:,1,1:]
        attention[name] = attn

    return hook


def get_mean_attention_map(attn, token, shape):
    attn = attn[:, :, token, 1:]
    attn = attn.unflatten(2, torch.Size([shape[1] // 16, shape[2] // 16])).float()
    attn = torch.nn.functional.interpolate(
        attn, size=shape[1:], mode="bicubic", align_corners=False
    ).squeeze(0)

    all_attn = torch.mean(attn, 0)

    return all_attn
