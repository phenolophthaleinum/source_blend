import numpy as np
from PIL import Image, ImageColor
import typer
import time
import srctools.vtf as vtf
import srctools.vpk as vpk
import srctools.vmt as vmt
from typing import Union, Optional
import gl_blendmodulate_bake as sh_blmod
import moderngl
import pathlib as p
import sys
import io


app = typer.Typer()


class BakeSession:
    def __init__(self):
        self.img_top: Optional[Image.Image] = None
        self.img_bottom: Optional[Image.Image] = None
        self.r_component: Optional[Image.Image] = None
        self.target_size: Optional[tuple[int, int]] = None
        self.output_image: Optional[Image.Image] = None
        self.gl = moderngl.create_standalone_context()
        self.quad = np.array([
                        # x, y, u, v
                        -1.0, -1.0, 0.0, 0.0,
                        1.0, -1.0, 1.0, 0.0,
                        -1.0,  1.0, 0.0, 1.0,
                        1.0,  1.0, 1.0, 1.0,
                    ], dtype='f4')


    def convert_colour_space(self, image: Image):
        """Convert image to RGBA, handling 16-bit and grayscale images efficiently."""
        im_arr = np.array(image)
        
        # Handle 16-bit images
        if im_arr.dtype == np.uint16 or im_arr.max() > 255:
            im_arr = (im_arr / 65535.0 * 255).astype(np.uint8)
            image = Image.fromarray(im_arr, mode=image.mode)
        
        # Convert to RGBA if needed (handles grayscale efficiently)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image


    def load_image(self, file_path: str) -> Image:
        start_time = time.time()
        try:
            img = Image.open(file_path)
        except Exception as e:
            try:
                if ".vpk:" in file_path:
                    vpk_path, _, internal_path = file_path.partition(".vpk:")
                    vpk_path = p.Path(f"{vpk_path}.vpk").resolve()
                    print(f"Loading VTF from VPK: {vpk_path}, internal path: {internal_path}", file=sys.stderr)
                    vpk_archive = vpk.VPK(vpk_path)
                    vpk_target_data = vpk_archive[internal_path].read()
                    import io
                    vtf_data = vtf.VTF.read(io.BytesIO(vpk_target_data))
                    print(f"Loaded VTF with size {vtf_data.width}x{vtf_data.height}", file=sys.stderr)
                    vtf_frame = vtf_data.get(frame=0, mipmap=0)
                    img = vtf_frame.to_PIL()
                else:
                    with open(file_path, 'rb') as f:
                        vtf_data = vtf.VTF.read(f)
                        print(f"Loaded VTF with size {vtf_data.width}x{vtf_data.height}", file=sys.stderr)
                        vtf_frame = vtf_data.get(frame=0, mipmap=0)
                        img = vtf_frame.to_PIL()
                # exit(1)
            except Exception as ve:
                print(f"Error - {file_path} is not a supported image: {ve}", file=sys.stderr)
                raise
        print(f"Loaded image in {time.time() - start_time:.4f} seconds", file=sys.stderr)
        start_time = time.time()
        img = self.convert_colour_space(img)
        print(f"Converted colour space in {time.time() - start_time:.4f} seconds", file=sys.stderr)
        return img


    def create_r_component(self, red: str) -> Image:
        """Create an image with only the red component from a color string."""
        self.target_size = self.target_size or self.img_top.size
        if red.startswith('#'):
            color = ImageColor.getcolor(red, "RGB")
            r_image = Image.new("RGB", (self.target_size[0], self.target_size[1]), color=(color[0], 0, 0))
            return r_image
        else:
            try:
                r_image = self.load_image(red)
            except Exception as e:
                print(f"Error loading red component image or wrong colour value. Defaulting to black: {e}", file=sys.stderr)
                r_image = Image.new("RGB", (self.target_size[0], self.target_size[1]), color=(0, 0, 0))
            return r_image
    
    def save_image(self, image: Image, file_path: str):
        file_path = p.Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.suffix.lower() == '.vtf':
            try:
                vtf_image = vtf.VTF(width=image.width, height=image.height, frames=1, fmt=vtf.ImageFormats.RGB888)
                vtf_image.get().copy_from(image.tobytes(), format=vtf.ImageFormats.RGB888)
                # vtf_image.save(str(file_path))
                with open(file_path, 'wb') as f:                                        
                    vtf_image.save(f)

            except Exception as e:
                print(f"Error saving VTF image: {e}", file=sys.stderr)
        else:
            try:
                image.save(file_path)
            except Exception as e:
                print(f"Error saving image: {e}", file=sys.stderr)
        print(f"Saved output image to {file_path}", file=sys.stderr)


class VMTSession:
    def __init__(self, base_texture: str,
                base_texture2: str,
                blendmodul_texture: str):
        self.base_texture: Optional[p.Path] = p.Path(base_texture).resolve()
        self.base_texture2: Optional[p.Path] = p.Path(base_texture2).resolve()
        self.blendmodul_texture: Optional[p.Path] = p.Path(blendmodul_texture).resolve()
        self._output_path: Optional[p.Path] = None
        self._relative_base_texture: Optional[str] = None
        self._relative_base_texture2: Optional[str] = None
        self._relative_blendmodul_texture: Optional[str] = None
        self.validate_paths()
    
    def validate_paths(self):
        for attr in self.__dict__:
            if attr.startswith('_'):
                continue
            path = getattr(self, attr)
            if path.suffix.lower() != ".vtf":
                raise ValueError(f"Texture {attr} does not have a .vtf extension: {path}")
            try:
                path_parts = path.as_posix().split("/")
                materials_index = path_parts.index('materials')
                relative_path = p.Path(*path_parts[materials_index + 1:]).with_suffix('')
                setattr(self, f"_relative_{attr}", str(relative_path))
            except (ValueError, IndexError):
                raise ValueError(f"Texture {attr} path should be included in 'materials/' directory: {path}")
            # path_str = str(path).replace('\\', '/')
            # if not 'materials/' in path_str.lower():
            #     raise ValueError(f"Texture {attr} path should be included in 'materials/': {path}")
            # relative_path = path_str.split('materials/', 1)[1]
            # relative_path = relative_path.rsplit('.', 1)[0]  # Remove extension
            # setattr(self, f"_relative_{attr}", relative_path)
        # return (self.base_texture.suffix.lower() == ".vtf" and
        #         self.base_texture2.suffix.lower() == ".vtf" and
        #         self.blendmodul_texture.suffix.lower() == ".vtf")
    def __str__(self) -> str:
        print_str = "VMT Session:\n"
        print_str += f"  Base Texture: {self.base_texture} (relative: {self._relative_base_texture})\n"
        print_str += f"  Base Texture 2: {self.base_texture2} (relative: {self._relative_base_texture2})\n"
        print_str += f"  Blendmodulate Texture: {self.blendmodul_texture} (relative: {self._relative_blendmodul_texture})\n"
        return print_str
    # def create_material(self) -> str:
    #     material = vmt.Material()


@app.command()
def show_img(
    file_path: str = typer.Option(None, "-p", "--path", help="Path to the image file. If not specified, will try reading from stdin.")):
    """Load and display image information."""
    session = BakeSession()
    if file_path is None:
        try:
            stdin_path = sys.stdin.read().strip()
            stdin_path = p.Path(stdin_path).resolve()
            file_path = str(stdin_path)
        except Exception as e:
            print(f"Incorrect image given: {e}")
            return
    img = session.load_image(file_path)
    print(f"Image size: {img.size}, mode: {img.mode}", file=sys.stderr)
    img.show()


@app.command()
def debug_red_component(
    top_img_path: str = typer.Option(..., "-t", "--top", help="Top image path"),
    red: str = typer.Option("#000000", "-r", "--red", help="Red component color or image path"),
    target_size: Union[None, str] = typer.Option(None, "-s", "--size", help="Target size as WIDTHxHEIGHT. None defaults to size of the top image.")):
    """Create and display the red component image."""
    session = BakeSession()
    top_img = session.load_image(top_img_path)
    session.img_top = top_img
    if target_size:
        width, height = map(int, target_size.lower().split('x'))
        session.target_size = (width, height)
    r_image = session.create_r_component(red)
    print(f"Red component image size: {r_image.size}, mode: {r_image.mode}", file=sys.stderr)
    r_image.show()


@app.command()
def bake_blendmodulate(
    top_img_path: str = typer.Option(..., "-t", "--top", help="Top image path"),
    bottom_img_path: str = typer.Option(..., "-b", "--bottom", help="Bottom image path"),
    red: str = typer.Option("#000000", "-r", "--red", help="Red component color or image path"),
    # output_path: str = typer.Option("output.png", "-o", "--output", help="Output image path"),
    target_size: Union[None, str] = typer.Option(None, "-s", "--size", help="Target size as WIDTHxHEIGHT. None defaults to size of the top image."),
    blend_mode: str = typer.Option("substract", "-m", "--mode", help="Blend mode to use."),
    output_path: Union[None, str] = typer.Option(None, "-o", "--output", help="Output image path. If not specified, the image will not be saved."),
    alpha: float = typer.Option(0.5, "-a", "--alpha", help="Alpha value for blending."),
    input_black: int = typer.Option(0, help="Input black level (0-255)."),
    input_white: int = typer.Option(255, help="Input white level (0-255)."),
    output_black: int = typer.Option(0, help="Output black level (0-255)."),
    output_white: int = typer.Option(255, help="Output white level (0-255).")
    ):
    """Bake blend modulate effect and save the output image."""
    start_time = time.time()
    session = BakeSession()
    top_img = session.load_image(top_img_path)
    bottom_img = session.load_image(bottom_img_path)
    session.img_top = top_img
    session.img_bottom = bottom_img
    if target_size:
        width, height = map(int, target_size.lower().split('x'))
        session.target_size = (width, height)
    r_image = session.create_r_component(red)
    session.r_component = r_image

    is_piped = not sys.stdout.isatty()
    
    print(session.img_top.size, session.img_bottom.size, session.r_component.size, file=sys.stderr)
    print(session.img_top.mode, session.img_bottom.mode, session.r_component.mode, file=sys.stderr)

    shader_cls = sh_blmod.BLEND_SHADERS.get(blend_mode.lower())
    if not shader_cls:
        print(f"Error: Unsupported blend mode '{blend_mode}'. Available modes: {list(sh_blmod.BLEND_SHADERS.keys())}", file=sys.stderr)
        return
    
    try:
        fbo = session.gl.framebuffer(
            color_attachments=[session.gl.texture((session.target_size[0], session.target_size[1]), 3)]
        )
        fbo.use()

        shader_program = session.gl.program(
            vertex_shader=shader_cls.vertex_shader,
            fragment_shader=shader_cls.fragment_shader
        )
        texture1 = session.gl.texture(top_img.size, 3, top_img.tobytes())
        texture2 = session.gl.texture(bottom_img.size, 3, bottom_img.tobytes())
        texture3 = session.gl.texture(r_image.size, 3, r_image.tobytes())
        texture1.use(location=0)
        texture2.use(location=1)
        texture3.use(location=2)
        shader_program[shader_cls.variables.topTexture] = 0
        shader_program[shader_cls.variables.bottomTexture] = 1
        shader_program[shader_cls.variables.rComponent] = 2
        shader_program[shader_cls.variables.alpha] = alpha
        shader_program[shader_cls.variables.input_black] = input_black / 255.0
        shader_program[shader_cls.variables.input_white] = input_white / 255.0
        shader_program[shader_cls.variables.output_black] = output_black / 255.0
        shader_program[shader_cls.variables.output_white] = output_white / 255.0

        vbo = session.gl.buffer(session.quad)
        vao = session.gl.simple_vertex_array(shader_program, 
                                            vbo,
                                            shader_cls.variables.in_vert,
                                            shader_cls.variables.in_text)
        session.gl.clear(0.0, 0.0, 0.0, 1.0)
        vao.render(moderngl.TRIANGLE_STRIP)
        data = fbo.read(components=3)
        blend_texture = Image.frombytes('RGB', (session.target_size[0], session.target_size[1]), data)
        print(f"Baked blend modulate (with texture loading) in {time.time() - start_time:.4f} seconds", file=sys.stderr)
        blend_texture.show()

        if output_path:
            session.save_image(blend_texture, output_path)
            if is_piped:
                #save output path to stdout
                print(output_path, file=sys.stdout)
    finally:
        fbo.release()
        shader_program.release()
        texture1.release()
        texture2.release()
        texture3.release()
        vbo.release()
        vao.release()


@app.command()
def create_vmt(
    base_texture: str = typer.Option(..., "-b", "--base", help="Base texture path (.vtf)"),
    base_texture2: str = typer.Option(..., "-b2", "--base2", help="Second base texture path (.vtf)"),
    blendmodul_texture: str = typer.Option(None, "-bm", "--blendmodul", help="Blend modulate texture path (.vtf)"),
    output_path: str = typer.Option("output.vmt", "-o", "--output", help="Output VMT file path")
    ):
    """Create a VMT file for blend modulate material."""
    session = VMTSession(base_texture, base_texture2, blendmodul_texture)
    print(session, file=sys.stderr)
    # material = vmt.Material()
    # material.set_shader("VertexLitGeneric")
    # material.set_keyvalue("$basetexture", session._relative_base_texture)
    # material.set_keyvalue("$basetexture2", session._relative_base_texture2)
    # material.set_keyvalue("$blendmodulatetexture", session._relative_blendmodul_texture)
    # material.set_keyvalue("$blendmodulatetype", "0")  # Assuming 0 is the correct type for standard blend modulate
    # output_vmt_str = material.to_string()
    # output_path_obj = p.Path(output_path)
    # output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    # with open(output_path_obj, 'w') as f:
    #     f.write(output_vmt_str)
    # print(f"Created VMT file at {output_path_obj}", file=sys.stderr)

if __name__ == "__main__":
    app()