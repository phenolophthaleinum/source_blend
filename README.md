## About
This was my private project to help me create easier and iterate much faster with the creation of blend modulate textures for Source Engine games. Decied to release it, maybe someone will find it helpful. 

If you got any idea to improve it please make a pull request according to Contributing section or just make an issue with an idea.

[Source blend pipeline](readme_static/source_blend_ffmpeg.mp4)

## Installation
You can download precompiled single file executable for Windows from the release section. Executable is compiled using [pyinstaller](https://github.com/pyinstaller/pyinstaller).

You can also download the source code and run it with your local python installation. Dependencies has to be installed - you can do it manually or create separate virual environment (recommended):
```
# create virtual environment
python -m venv path/to/venv

# activate environment - Windows
.\path\to\venv\Scripts\activate

# activate environment - Linux
source path/to/venv/bin/activate

# install dependencies
pip install -r requirements.txt
```
## Usage
### bake-blendmodulate
Create blend modulate texture. Images can be passed in any image format supported by Pillow, or in VTF. VTF from VPK packages can be passed directly by giving path to VPK (_dir) and relative VTF path connected with `:`. Example:
```
# exec version
.\source_blend.exe bake-blendmodulate -t top_texture.png -b X:\Steam\steamapps\common\GarrysMod\garrysmod\garrysmod_dir.vpk:materials/brick/brickwall003a.vtf -r "#111111" -s 2048x2048 -a 0.1 -m subtract -o X:\Steam\steamapps\common\GarrysMod\garrysmod\materials\custom\blendmodulate.vtf
```
`-r` is the colour or image passed to the red channel. `-s` is desired output image size. `-a` is blending power between top and bottom texture. `-m` is type of blending to be used. 

If you would like immediately preview created image, use `-p`.

For more details about `bake-blendmodulate`, please refer to the help page:
```
.\source_blend.exe bake-blendmodulate --help
```

### create-vmt
You can quickly create VMT to inspect your blend in the Hammer faster. The required inputs are the same as for the WorldVertexTransition material, though limited to basic parameters:
- basetexture
- bumpmap
- basetexture2
- bumpmap2
- blendmodulatetexture
  
Paths to images can be relative, since they need to make sense for the engine. Full paths will be stripped to relative paths (relative to `materials/`). Path name must contain `materials/`.

**Output path must be a full path.**

Normalmaps are optional.

Example:
```
.\source_blend.exe create-vmt -b E:\Steam\steamapps\common\GarrysMod\garrysmod\materials\custom\basetexture.vtf -b2 materials\custom\basetexture2.vtf -bl materials\custom\blendmodulate.vtf -o X:\Steam\steamapps\common\GarrysMod\garrysmod\materials\custom\generated_blend.vmt -bm materials\custom\normalmap.vtf -bm2 E:\Steam\steamapps\common\GarrysMod\garrysmod\materials\custom\normalmap2.vtf
```
For more details about `create-vmt`, please refer to the help page:
```
.\source_blend.exe create-vmt --help
```

### Full pipeline
You can pipe the result of the bake to `create-vmt` for fully automated process. Engine must have the texture on the disk, so the baked blendmodulate texture needs to be saved (`-o` parameter). When piping, `create-vmt` does not need blendmodulate texture path specified by `-bl` parameter.

Example:
```
.\source_blend.exe bake-blendmodulate -t top_texture.png -b X:\Steam\steamapps\common\GarrysMod\garrysmod\garrysmod_dir.vpk:materials/brick/brickwall003a.vtf -r "#111111" -s 2048x2048 -a 0.1 -m subtract -o X:\Steam\steamapps\common\GarrysMod\garrysmod\materials\custom\blendmodulate.vtf | .\source_blend.exe create-vmt -b E:\Steam\steamapps\common\GarrysMod\garrysmod\materials\custom\basetexture.vtf -b2 materials\custom\basetexture2.vtf -o X:\Steam\steamapps\common\GarrysMod\garrysmod\materials\custom\generated_blend.vmt -bm materials\custom\normalmap.vtf -bm2 E:\Steam\steamapps\common\GarrysMod\garrysmod\materials\custom\normalmap2.vtf
```

## Contributing
If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/featureName`)
3. Commit your Changes (`git commit -m 'Add something'`)
4. Push to the Branch (`git push origin feature/featureName`)
5. Open a Pull Request

