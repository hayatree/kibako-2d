# Licensing policy

kibako aims to release models whose outputs can be used in commercial games. A model is only released when all three conditions below hold.

## 1. Base model

The base model must be licensed for commercial use.

| Base model | License | Allowed |
|---|---|---|
| FLUX.2 [klein] 4B | Apache-2.0 | yes |
| Z-Image | Apache-2.0 | yes |
| Qwen-Image (up to 2512, Edit-2511, Layered) | Apache-2.0 | yes |
| FLUX.2 [klein] 9B, FLUX.1 [dev] | FLUX Non-Commercial | no |
| Qwen-Image 2.1 | Qwen Research License | no |

Licenses are re-checked against the official model card before every release.

## 2. Training data

Only the following licenses are accepted (enforced by `kibako.manifest`):

- `CC0-1.0`
- `LicenseRef-own-work`: original work by the maintainers, released under CC0

Every training image has a manifest entry with its SHA-256 hash, source, source URL and license. The manifest is committed to this repository and published with the dataset, so the provenance of every image can be audited.

## 3. Released weights

LoRA weights are released under Apache-2.0.

## Disclaimer

This is not legal advice. Whether AI-generated images are protected by copyright differs by jurisdiction. Avoid prompts that name existing characters, brands or other third-party IP.
