"""Export the pretrained CoMER handwriting model to two ONNX graphs.

encoder.onnx : image [1, 1, H, W] (white ink = 1.0 on black)      -> feature [1, h, w, 256]
decoder.onnx : feature [B, h, w, 256], tokens [B, L] (int64)        -> log-probs of the next token [B, V]

The inference path is re-implemented here without padding masks and einops so
that every shape stays dynamic in the exported graph. The script checks the
re-implementation against the original model and the ONNX graphs against
PyTorch, then measures the expression recognition rate on CROHME 2014.

usage: python export_onnx.py <CoMER repo dir> <out dir> [n_eval]
"""

import math
import os
import sys
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

COMER = os.path.abspath(sys.argv[1])
OUT = os.path.abspath(sys.argv[2])
N_EVAL = int(sys.argv[3]) if len(sys.argv) > 3 else 300
sys.path.insert(0, COMER)

from comer.datamodule import vocab  # noqa: E402
from comer.model.comer import CoMER  # noqa: E402

CKPT = os.path.join(COMER, "lightning_logs/version_0/checkpoints/epoch=151-step=57151-val_ExpRate=0.6365.ckpt")


def load_model():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    hp = dict(ckpt["hyper_parameters"])
    model = CoMER(
        d_model=hp["d_model"],
        growth_rate=hp["growth_rate"],
        num_layers=hp["num_layers"],
        nhead=hp["nhead"],
        num_decoder_layers=hp["num_decoder_layers"],
        dim_feedforward=hp["dim_feedforward"],
        dropout=hp["dropout"],
        dc=hp["dc"],
        cross_coverage=hp["cross_coverage"],
        self_coverage=hp["self_coverage"],
    )
    sd = {k[len("comer_model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("comer_model.")}
    model.load_state_dict(sd)
    model.eval()
    return model


class EncoderExport(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        d_model = enc.feature_proj.out_channels
        half = d_model // 2
        dim_t = torch.arange(0, half, 2, dtype=torch.float)
        self.register_buffer("inv_freq", 1.0 / (10000.0 ** (dim_t / half)))
        self.scale = 2 * math.pi

    @staticmethod
    def transition(t, x):
        x = F.relu(t.bn1(t.conv1(x)))
        num = F.avg_pool2d(F.pad(x, (0, 1, 0, 1)), 2)
        den = F.avg_pool2d(F.pad(torch.ones_like(x[:, :1]), (0, 1, 0, 1)), 2)
        return num / den

    def forward(self, img):
        d = self.enc.model
        out = d.conv1(img)
        out = d.norm1(out)
        out = F.relu(out)
        # ceil_mode pooling written with explicit padding: ONNX Runtime treats the
        # clipped edge windows of ceil_mode AveragePool differently from PyTorch.
        out = F.max_pool2d(F.pad(out, (0, 1, 0, 1)), 2)  # values are >= 0 after ReLU
        out = d.dense1(out)
        out = self.transition(d.trans1, out)
        out = d.dense2(out)
        out = self.transition(d.trans2, out)
        out = d.dense3(out)
        out = d.post_norm(out)
        f = self.enc.feature_proj(out).permute(0, 2, 3, 1)  # [b, h, w, d]

        ones = torch.ones_like(f[..., 0])  # [b, h, w]
        y_embed = ones.cumsum(1)
        x_embed = ones.cumsum(2)
        y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale
        pos_x = x_embed.unsqueeze(-1) * self.inv_freq
        pos_y = y_embed.unsqueeze(-1) * self.inv_freq
        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=4).flatten(3)
        f = f + torch.cat((pos_x, pos_y), dim=3)
        return self.enc.norm(f)


class DecoderExport(nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec
        self.nhead = dec.model.layers[0].self_attn.num_heads
        self.d_model = dec.model.layers[0].self_attn.embed_dim
        self.hd = self.d_model // self.nhead
        self.arm = dec.model.arm

    def heads(self, x):  # [b, n, d] -> [b, H, n, hd]
        return x.unflatten(-1, (self.nhead, self.hd)).transpose(1, 2)

    def merge(self, x):  # [b, H, n, hd] -> [b, n, d]
        return x.transpose(1, 2).flatten(2)

    def arm_bias(self, prev_attn, curr_attn, feature):
        """Attention refinement; attn tensors are [b, H, l, s]."""
        arm = self.arm
        h = feature.shape[1]
        w = feature.shape[2]
        attns = torch.cat((prev_attn, curr_attn), dim=1)  # [b, 2H, l, s]
        attns = attns.cumsum(dim=2) - attns
        b2 = attns.permute(0, 2, 1, 3)  # [b, l, 2H, s]
        x = b2.reshape(-1, attns.shape[1], h, w)  # [(b l), 2H, h, w]
        cov = arm.act(arm.conv(x))
        cov = arm.proj(cov)
        bn = arm.post_norm.bn
        cov = F.batch_norm(cov, bn.running_mean, bn.running_var, bn.weight, bn.bias, False, 0.0, bn.eps)
        cov = cov.flatten(2)  # [(b l), H, s]
        cov = cov.reshape(curr_attn.shape[0], curr_attn.shape[2], self.nhead, -1)  # [b, l, H, s]
        return cov.permute(0, 2, 1, 3)

    def forward(self, feature, tgt):
        dec = self.dec
        pos = torch.ones_like(tgt).cumsum(1) - 1  # [b, l]
        x = dec.word_embed(tgt)
        x = x + F.embedding(pos, dec.pos_enc.pe)
        x = dec.norm(x)

        memory = feature.flatten(1, 2)  # [b, s, d]
        p = pos[0]
        causal = p.unsqueeze(0) > p.unsqueeze(1)  # [l, l] True above the diagonal
        scaling = float(self.hd) ** -0.5

        prev = None
        n_layers = len(dec.model.layers)
        for i, layer in enumerate(dec.model.layers):
            # self attention
            sa = layer.self_attn
            q, k, v = F.linear(x, sa.in_proj_weight, sa.in_proj_bias).chunk(3, dim=-1)
            q = self.heads(q * scaling)
            k, v = self.heads(k), self.heads(v)
            dots = q @ k.transpose(-1, -2)
            dots = dots.masked_fill(causal, float("-inf"))
            att = F.softmax(dots, dim=-1)
            out = sa.out_proj(self.merge(att @ v))
            x = layer.norm1(x + out)

            # cross attention with coverage refinement
            ca = layer.multihead_attn
            d = self.d_model
            q = F.linear(x, ca.in_proj_weight[:d], ca.in_proj_bias[:d])
            k, v = F.linear(memory, ca.in_proj_weight[d:], ca.in_proj_bias[d:]).chunk(2, dim=-1)
            q = self.heads(q * scaling)
            k, v = self.heads(k), self.heads(v)
            dots = q @ k.transpose(-1, -2)  # [b, H, l, s]
            att = F.softmax(dots, dim=-1)
            if prev is not None and self.arm is not None:
                dots = dots - self.arm_bias(prev, att, feature)
                att = F.softmax(dots, dim=-1)
            out = ca.out_proj(self.merge(att @ v))
            x = layer.norm2(x + out)

            x = layer.norm3(x + layer.linear2(F.relu(layer.linear1(x))))
            if i != n_layers - 1:
                prev = att

        logits = dec.proj(x[:, -1, :])
        return F.log_softmax(logits, dim=-1)


def load_images(n):
    with zipfile.ZipFile(os.path.join(COMER, "data.zip")) as zf:
        lines = zf.read("data/2014/caption.txt").decode().strip().split("\n")
        data = []
        for line in lines[:n]:
            parts = line.strip().split()
            with zf.open("data/2014/img/%s.bmp" % parts[0]) as f:
                img = np.array(Image.open(f).convert("L"), dtype=np.float32) / 255.0
            data.append((parts[0], img, parts[1:]))
    return data


def scale_to_range(img):
    h, w = img.shape
    r = min(256 / h, 1024 / w)
    if r < 1.0:
        im = Image.fromarray((img * 255).astype(np.uint8)).resize((max(1, int(w * r)), max(1, int(h * r))), Image.BILINEAR)
        return np.array(im, dtype=np.float32) / 255.0
    return img


def beam_search(enc_fn, dec_fn, img, beam=5, max_len=120):
    """Left-to-right beam search; the Kotlin implementation mirrors this function."""
    feat = enc_fn(img[None, None])
    beams = [([vocab.SOS_IDX], 0.0)]
    done = []
    for _ in range(max_len):
        tokens = np.array([b[0] for b in beams], dtype=np.int64)
        feats = np.repeat(feat, len(beams), axis=0)
        logp = dec_fn(feats, tokens)
        cand = []
        for bi, (seq, score) in enumerate(beams):
            top = np.argsort(-logp[bi])[:beam]
            for t in top:
                cand.append((seq + [int(t)], score + float(logp[bi, t])))
        cand.sort(key=lambda c: -c[1])
        beams = []
        for seq, score in cand:
            if seq[-1] == vocab.EOS_IDX:
                done.append((seq[1:-1], score / (len(seq) - 1)))
            else:
                beams.append((seq, score))
            if len(beams) == beam:
                break
        if len(done) >= beam or not beams:
            break
        best_done = max((d[1] for d in done), default=-1e9)
        if beams and all(b[1] / len(b[0]) < best_done for b in beams) and len(done) > 0:
            break
    if not done:
        done = [(b[0][1:], b[1] / len(b[0])) for b in beams]
    best = max(done, key=lambda d: d[1])
    return [vocab.idx2word[i] for i in best[0]]


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    model = load_model()
    enc = EncoderExport(model.encoder).eval()
    dec = DecoderExport(model.decoder).eval()

    data = load_images(max(N_EVAL, 20))

    # 1) re-implementation == original model
    with torch.no_grad():
        for name, img, cap in data[:8]:
            x = torch.from_numpy(scale_to_range(img))[None, None]
            mask = torch.zeros(1, x.shape[2], x.shape[3], dtype=torch.bool)
            f_ref, m_ref = model.encoder(x, mask)
            f_new = enc(x)
            assert f_ref.shape == f_new.shape, (f_ref.shape, f_new.shape)
            tgt = torch.tensor([[vocab.SOS_IDX] + vocab.words2indices(cap)])
            ref = F.log_softmax(model.decoder(f_ref, m_ref, tgt)[:, -1], dim=-1)
            new = dec(f_new, tgt)
            err_f = (f_ref - f_new).abs().max().item()
            err_d = (ref - new).abs().max().item()
            print("check %s feat_err=%.2e dec_err=%.2e" % (name, err_f, err_d))
            assert err_f < 1e-3 and err_d < 1e-3

    # 2) export
    x = torch.rand(1, 1, 80, 240)
    with torch.no_grad():
        feat = enc(x)
    tgt = torch.tensor([[1, 20, 30], [1, 21, 31]])
    feat2 = feat.repeat(2, 1, 1, 1)
    enc_path = os.path.join(OUT, "encoder.onnx")
    dec_path = os.path.join(OUT, "decoder.onnx")
    torch.onnx.export(
        enc, (x,), enc_path, input_names=["image"], output_names=["feature"],
        dynamic_axes={"image": {2: "H", 3: "W"}, "feature": {1: "h", 2: "w"}},
        opset_version=17, dynamo=False,
    )
    torch.onnx.export(
        dec, (feat2, tgt), dec_path, input_names=["feature", "tokens"], output_names=["logp"],
        dynamic_axes={"feature": {0: "B", 1: "h", 2: "w"}, "tokens": {0: "B", 1: "L"}, "logp": {0: "B"}},
        opset_version=17, dynamo=False,
    )

    import onnxruntime as ort

    so = ort.SessionOptions()
    s_enc = ort.InferenceSession(enc_path, so, providers=["CPUExecutionProvider"])
    s_dec = ort.InferenceSession(dec_path, so, providers=["CPUExecutionProvider"])

    def enc_fn(a):
        return s_enc.run(None, {"image": a.astype(np.float32)})[0]

    def dec_fn(f, t):
        return s_dec.run(None, {"feature": f, "tokens": t})[0]

    # 3) ONNX == PyTorch for several sizes and batch sizes
    with torch.no_grad():
        for name, img, cap in data[:6]:
            a = scale_to_range(img)
            f_t = enc(torch.from_numpy(a)[None, None]).numpy()
            f_o = enc_fn(a[None, None])
            seq = [vocab.SOS_IDX] + vocab.words2indices(cap)
            t = np.array([seq, seq[:1] + seq[1:][::-1]], dtype=np.int64)
            d_t = dec(torch.from_numpy(np.repeat(f_t, 2, 0)), torch.from_numpy(t)).numpy()
            d_o = dec_fn(np.repeat(f_o, 2, 0), t)
            print("onnx %s %s feat_err=%.2e dec_err=%.2e" % (name, a.shape, np.abs(f_t - f_o).max(), np.abs(d_t - d_o).max()))
            assert np.abs(f_t - f_o).max() < 1e-3 and np.abs(d_t - d_o).max() < 1e-3

    # 4) recognition rate with the ONNX graphs
    correct = 0
    for k, (name, img, cap) in enumerate(data[:N_EVAL]):
        pred = beam_search(enc_fn, dec_fn, scale_to_range(img))
        ok = pred == cap
        correct += ok
        if k < 15:
            print("%-5s %s | %s" % ("OK" if ok else "MISS", " ".join(cap), " ".join(pred)))
    print("ExpRate on %d CROHME-2014 images: %.1f%%" % (N_EVAL, 100.0 * correct / N_EVAL))

    with open(os.path.join(OUT, "vocab.txt"), "w") as f:
        for i in range(len(vocab)):
            f.write(vocab.idx2word[i] + "\n")
    for p in (enc_path, dec_path):
        print(p, os.path.getsize(p) // 1024, "KiB")


if __name__ == "__main__":
    main()
