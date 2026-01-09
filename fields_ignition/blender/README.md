# Tomato Plant Generator for Blender

A procedural tomato plant generator using Blender's Python API. Generates realistic 3D tomato plant models with stems, leaves, flowers, and fruits.

## Requirements

- Blender 4.0 or later
- Python 3.10+ (included with Blender)

## Installation

### macOS

1. **Install Blender**

   Using Homebrew:
   ```bash
   brew install --cask blender
   ```

   Or download from [blender.org](https://www.blender.org/download/).

2. **Verify installation**
   ```bash
   /Applications/Blender.app/Contents/MacOS/Blender --version
   ```

### Linux (Ubuntu/Debian)

1. **Install Blender**

   Using apt:
   ```bash
   sudo apt update
   sudo apt install blender
   ```

   Or using snap (recommended for latest version):
   ```bash
   sudo snap install blender --classic
   ```

   Or download from [blender.org](https://www.blender.org/download/).

2. **Verify installation**
   ```bash
   blender --version
   ```

## Usage

### Basic Usage

Generate a tomato plant with default settings:

```bash
# macOS
/Applications/Blender.app/Contents/MacOS/Blender tomato_gen.blend --background --python tomato_gen.py -- --model_dir ./output --seed 12345

# Linux
blender tomato_gen.blend --background --python tomato_gen.py -- --model_dir ./output --seed 12345 --format fbx --with-textures
```

### Export Formats

```bash
# FBX with embedded textures
blender tomato_gen.blend --background --python tomato_gen.py -- \
    --model_dir ./output \
    --format fbx \
    --with-textures

# glTF Binary (.glb) - Recommended for game engines
blender tomato_gen.blend --background --python tomato_gen.py -- \
    --model_dir ./output \
    --export-format glb \
    --with-textures

# USD - For NVIDIA Isaac Sim / Omniverse
blender tomato_gen.blend --background --python tomato_gen.py -- \
    --model_dir ./output \
    --export-format usd \
    --with-textures
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --model_dir` | Output directory | `generated/test_tomato` |
| `-s, --seed` | Random seed for reproducibility | Random |
| `-f, --format` | Export format: `dae`, `fbx` | `dae` |
| `--export-format` | Extended formats: `dae`, `fbx`, `glb`, `usd` | - |
| `--with-textures` | Embed textures in export | Off |
| `--alpha-clip` | Remove faces with alpha below threshold (0.0-1.0) | - |
| `--alpha-clip-mode` | Face removal mode: `any`, `all`, `majority` | `majority` |
| `--remesh-alpha` | Remesh to follow alpha contours | Off |

## Output Structure

```
output_dir/
├── meshes/
│   └── tomato.{dae,fbx,glb,usd}   # 3D model with submeshes
├── markers.json                    # Fruit positions (ground truth)
```

### Submeshes

The exported model contains named submeshes:
- `Branch1` - Main stem and sub-stems
- `Leaf1`, `Leaf2` - Leaf geometry
- `Fruit1` - `Fruit4` - Fruit geometry
- `Blossom1` - `Blossom3` - Flower geometry

## Handling Transparency in Isaac Sim

When importing into NVIDIA Isaac Sim, transparent textures may appear black. To fix this:

1. Import the GLB/USD file into Isaac Sim
2. Select the material in the Property Panel
3. Configure OmniPBR material settings:
   - **Enable Opacity**: ON
   - **Enable Opacity Mapping**: ON
   - **Opacity Texture**: Select the texture with alpha channel
   - **Opacity Threshold**: `0` (blend) or `0.5` (cutout)

See [Isaac Sim Transparency Documentation](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/templates/parameters/OmniPBR_Opacity.html) for details.

## Algorithm

See [ALGORITHM.md](ALGORITHM.md) for detailed documentation of the generation algorithm.

### Overview

The generator uses a hybrid approach:
1. **Procedural geometry** - Main stem is generated mathematically
2. **Prefab instancing** - Leaves, flowers, and fruits are duplicated from pre-built meshes

### Reproducibility

All random operations use NumPy's seeded random number generator:
```bash
--seed 12345  # Same seed = identical plant
```

## Textures

Textures are located in the `textures/` directory:

| File | Description |
|------|-------------|
| `brn1.png` | Branch/stem texture |
| `lef1.png`, `lef2.png` | Leaf textures |
| `frt1-4.png` | Fruit textures (different ripeness) |
| `blo1-3.png` | Blossom/flower textures |

## Examples

### Generate multiple plants with different seeds

```bash
for i in {1..10}; do
    blender tomato_gen.blend --background --python tomato_gen.py -- \
        --model_dir ./output/plant_$i \
        --seed $i \
        --export-format glb \
        --with-textures
done
```

### Generate for Gazebo simulation

```bash
blender tomato_gen.blend --background --python tomato_gen.py -- \
    --model_dir ./output \
    --format dae \
    --seed 42
```

## Troubleshooting

### "blender: command not found" on macOS

Add Blender to your PATH or use the full path:
```bash
alias blender="/Applications/Blender.app/Contents/MacOS/Blender"
```

### Textures not appearing in export

Make sure to use `--with-textures` flag:
```bash
blender tomato_gen.blend --background --python tomato_gen.py -- \
    --export-format glb \
    --with-textures
```

### Black areas in transparency (Isaac Sim)

This is a material configuration issue in Isaac Sim, not a mesh problem. See the "Handling Transparency in Isaac Sim" section above.

## License

See the repository root for license information.
