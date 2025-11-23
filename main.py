import numpy as np
from PIL import Image, ImageColor
import typer
import time
import srctools.vtf as vtf
import srctools.vpk as vpk
from typing import Union, Optional
import gl_blendmodulate_bake as sh_blmod
import moderngl
import pathlib as p


app = typer.Typer()


class Session:
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
                    print(f"Loading VTF from VPK: {vpk_path}, internal path: {internal_path}")
                    vpk_archive = vpk.VPK(vpk_path)
                    vpk_target_data = vpk_archive[internal_path].read()
                    import io
                    vtf_data = vtf.VTF.read(io.BytesIO(vpk_target_data))
                    print(f"Loaded VTF with size {vtf_data.width}x{vtf_data.height}")
                    vtf_frame = vtf_data.get(frame=0, mipmap=0)
                    img = vtf_frame.to_PIL()
                else:
                    with open(file_path, 'rb') as f:
                        vtf_data = vtf.VTF.read(f)
                        print(f"Loaded VTF with size {vtf_data.width}x{vtf_data.height}")
                        vtf_frame = vtf_data.get(frame=0, mipmap=0)
                        img = vtf_frame.to_PIL()
                # exit(1)
            except Exception as ve:
                print(f"Error - {file_path} is not a supported image: {ve}")
                raise
        print(f"Loaded image in {time.time() - start_time:.4f} seconds")
        start_time = time.time()
        img = self.convert_colour_space(img)
        print(f"Converted colour space in {time.time() - start_time:.4f} seconds")
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
                print(f"Error loading red component image or wrong colour value. Defaulting to black: {e}")
                r_image = Image.new("RGB", (self.target_size[0], self.target_size[1]), color=(0, 0, 0))
            return r_image


@app.command()
def show_img(
    file_path: str = typer.Option(..., "-p", "--path", help="Path to the image file")):
    """Load and display image information."""
    session = Session()
    img = session.load_image(file_path)
    print(f"Image size: {img.size}, mode: {img.mode}")
    img.show()

@app.command()
def debug_red_component(
    top_img_path: str = typer.Option(..., "-t", "--top", help="Top image path"),
    red: str = typer.Option("#000000", "-r", "--red", help="Red component color or image path"),
    target_size: Union[None, str] = typer.Option(None, "-s", "--size", help="Target size as WIDTHxHEIGHT. None defaults to size of the top image.")):
    """Create and display the red component image."""
    session = Session()
    top_img = session.load_image(top_img_path)
    session.img_top = top_img
    if target_size:
        width, height = map(int, target_size.lower().split('x'))
        session.target_size = (width, height)
    r_image = session.create_r_component(red)
    print(f"Red component image size: {r_image.size}, mode: {r_image.mode}")
    r_image.show()


@app.command()
def bake_blendmodulate(
    top_img_path: str = typer.Option(..., "-t", "--top", help="Top image path"),
    bottom_img_path: str = typer.Option(..., "-b", "--bottom", help="Bottom image path"),
    red: str = typer.Option("#000000", "-r", "--red", help="Red component color or image path"),
    # output_path: str = typer.Option("output.png", "-o", "--output", help="Output image path"),
    target_size: Union[None, str] = typer.Option(None, "-s", "--size", help="Target size as WIDTHxHEIGHT. None defaults to size of the top image."),
    alpha: float = typer.Option(0.5, "-a", "--alpha", help="Alpha value for blending."),
    input_black: int = typer.Option(0, help="Input black level (0-255)."),
    input_white: int = typer.Option(255, help="Input white level (0-255)."),
    output_black: int = typer.Option(0, help="Output black level (0-255)."),
    output_white: int = typer.Option(255, help="Output white level (0-255).")
    ):
    """Bake blend modulate effect and save the output image."""
    start_time = time.time()
    session = Session()
    top_img = session.load_image(top_img_path)
    bottom_img = session.load_image(bottom_img_path)
    session.img_top = top_img
    session.img_bottom = bottom_img
    if target_size:
        width, height = map(int, target_size.lower().split('x'))
        session.target_size = (width, height)
    r_image = session.create_r_component(red)
    session.r_component = r_image
    
    print(session.img_top.size, session.img_bottom.size, session.r_component.size)
    print(session.img_top.mode, session.img_bottom.mode, session.r_component.mode)

    fbo = session.gl.framebuffer(
        color_attachments=[session.gl.texture((session.target_size[0], session.target_size[1]), 3)]
    )
    fbo.use()
    shader_program = session.gl.program(
        vertex_shader=sh_blmod.Shader.vertex_shader,
        fragment_shader=sh_blmod.Shader.fragment_shader
    )
    texture1 = session.gl.texture(top_img.size, 3, top_img.tobytes())
    texture2 = session.gl.texture(bottom_img.size, 3, bottom_img.tobytes())
    texture3 = session.gl.texture(r_image.size, 3, r_image.tobytes())
    texture1.use(location=0)
    texture2.use(location=1)
    texture3.use(location=2)
    shader_program[sh_blmod.Shader.variables.topTexture] = 0
    shader_program[sh_blmod.Shader.variables.bottomTexture] = 1
    shader_program[sh_blmod.Shader.variables.rComponent] = 2
    shader_program[sh_blmod.Shader.variables.alpha] = alpha
    shader_program[sh_blmod.Shader.variables.input_black] = input_black / 255.0
    shader_program[sh_blmod.Shader.variables.input_white] = input_white / 255.0
    shader_program[sh_blmod.Shader.variables.output_black] = output_black / 255.0
    shader_program[sh_blmod.Shader.variables.output_white] = output_white / 255.0

    vbo = session.gl.buffer(session.quad)
    vao = session.gl.simple_vertex_array(shader_program, 
                                         vbo,
                                         sh_blmod.Shader.variables.in_vert,
                                         sh_blmod.Shader.variables.in_text)
    session.gl.clear(0.0, 0.0, 0.0, 1.0)
    vao.render(moderngl.TRIANGLE_STRIP)
    data = fbo.read(components=3)
    blend_texture = Image.frombytes('RGB', (session.target_size[0], session.target_size[1]), data)
    print(f"Baked blend modulate (with texture loading) in {time.time() - start_time:.4f} seconds")
    blend_texture.show()


if __name__ == "__main__":
    app()