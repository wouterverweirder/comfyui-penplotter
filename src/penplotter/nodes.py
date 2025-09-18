from inspect import cleandoc
import numpy as np
import tempfile
import subprocess
from pathlib import Path
from server import PromptServer
from PIL import Image
import comfy.model_management

# Global variable to track the currently running subprocess for interruption
_current_subprocess = None

class ImageToCenterline:
    CATEGORY = "Pen Plotter"
    
    @classmethod    
    def INPUT_TYPES(s):
        return { 
            "required": { 
                "image": ("IMAGE",),
            },
            "optional": {
                "optimize_exhaustive": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("svg_string",)
    FUNCTION = "convert_to_centerline"

    def tensor_to_pil(self, tensor):
        """Convert ComfyUI tensor to PIL Image."""
        # ComfyUI images are in format [batch, height, width, channels] with values 0-1
        if len(tensor.shape) == 4:
            tensor = tensor[0]  # Take first image from batch
        
        # Convert from 0-1 to 0-255 and to numpy
        numpy_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
        
        # Convert to PIL Image
        if numpy_image.shape[-1] == 3:  # RGB
            return Image.fromarray(numpy_image, 'RGB')
        elif numpy_image.shape[-1] == 1:  # Grayscale
            return Image.fromarray(numpy_image.squeeze(-1), 'L')
        else:
            # Handle RGBA or other formats
            return Image.fromarray(numpy_image, 'RGBA')

    def convert_to_ppm(self, pil_image, output_path):
        """Convert PIL image to PPM format."""
        try:
            # Convert to RGB if necessary (PPM doesn't support transparency)
            if pil_image.mode in ('RGBA', 'LA'):
                # Create a white background
                background = Image.new('RGB', pil_image.size, (255, 255, 255))
                if pil_image.mode == 'RGBA':
                    background.paste(pil_image, mask=pil_image.split()[-1])  # Use alpha channel as mask
                else:
                    background.paste(pil_image, mask=pil_image.split()[-1])  # Use alpha channel as mask
                pil_image = background
            elif pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Save as PPM
            pil_image.save(output_path, 'PPM')
            return True
        except Exception as e:
            print(f"Error converting image to PPM: {e}")
            return False

    def run_raster_retrace(self, ppm_path, svg_path, optimize_exhaustive=True):
        """Run the raster-retrace command on a PPM file to generate SVG."""
        global _current_subprocess
        try:
            # Build command with proper flags
            cmd = [
                'raster-retrace',
                '-i', str(ppm_path),
                '-o', str(svg_path),
                '-m', 'CENTER'
            ]
            
            if optimize_exhaustive:
                cmd.append('--optimize-exhaustive')
            
            _current_subprocess = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = _current_subprocess.communicate()
            
            if _current_subprocess.returncode == 0:
                _current_subprocess = None
                return True
            else:
                print(f"Error running raster-retrace: {stderr}")
                _current_subprocess = None
                return False
        except FileNotFoundError:
            print("Error: raster-retrace command not found. Please ensure it's installed and in your PATH.")
            _current_subprocess = None
            return False
        except Exception as e:
            print(f"Error running raster-retrace: {e}")
            _current_subprocess = None
            return False

    def convert_to_centerline(self, image, optimize_exhaustive=True):
        """Convert image tensor to centerline SVG string."""
        if Image is None:
            return ("Error: PIL (Pillow) is required but not installed. Please install it with 'pip install Pillow'",)
            
        try:
            # Convert tensor to PIL Image
            pil_image = self.tensor_to_pil(image)
            
            # Create temporary files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                ppm_path = temp_path / "temp_image.ppm"
                svg_path = temp_path / "temp_output.svg"
                
                # Convert to PPM
                if not self.convert_to_ppm(pil_image, str(ppm_path)):
                    return ("Error: Failed to convert image to PPM format",)
                
                # Run raster-retrace
                if not self.run_raster_retrace(ppm_path, svg_path, optimize_exhaustive):
                    return ("Error: Failed to run raster-retrace conversion",)
                
                # Read the SVG file and return as string
                if svg_path.exists():
                    with open(svg_path, 'r', encoding='utf-8') as f:
                        svg_content = f.read()
                    return (svg_content,)
                else:
                    return ("Error: SVG file was not created",)
                    
        except Exception as e:
            return (f"Error: {str(e)}",)


class OptimizeSVG:
    CATEGORY = "Pen Plotter"
    
    @classmethod    
    def INPUT_TYPES(s):
        return { 
            "required": { 
                "svg_string": ("STRING", {"multiline": True}),
            },
            "optional": {
                "fit_to_margins": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "landscape": ("BOOLEAN", {"default": True}),
                "linemerge": ("BOOLEAN", {"default": True}),
                "tolerance": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 5.0, "step": 0.01}),
                "linesort": ("BOOLEAN", {"default": True}),
                "reloop": ("BOOLEAN", {"default": True}),
                "linesimplify": ("BOOLEAN", {"default": True}),
                "page_width": ("FLOAT", {"default": 100.0, "min": 10.0, "max": 500.0, "step": 1.0}),
                "page_height": ("FLOAT", {"default": 148.0, "min": 10.0, "max": 500.0, "step": 1.0}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("optimized_svg",)
    FUNCTION = "optimize_svg"

    def run_vpype_command(self, input_path, output_path, fit_to_margins, landscape, linemerge, 
                         tolerance, linesort, reloop, linesimplify, page_width, page_height):
        """Run vpype command with specified parameters."""
        global _current_subprocess
        try:
            # Build the vpype command
            cmd = ['vpype', 'read', str(input_path)]
            
            # Add layout command with fit-to-margins and landscape options
            layout_cmd = ['layout', f'--fit-to-margins', f'{fit_to_margins}mm']
            if landscape:
                layout_cmd.extend(['--landscape', f'{page_width}x{page_height}mm'])
            else:
                layout_cmd.append(f'{page_width}x{page_height}mm')
            cmd.extend(layout_cmd)
            
            # Add linemerge if enabled
            if linemerge:
                cmd.extend(['linemerge', '--tolerance', f'{tolerance}mm'])
            
            # Add linesort if enabled
            if linesort:
                cmd.append('linesort')
            
            # Add reloop if enabled
            if reloop:
                cmd.append('reloop')
            
            # Add linesimplify if enabled
            if linesimplify:
                cmd.append('linesimplify')
            
            # Add write command with page size
            if landscape:
                cmd.extend(['write', '--page-size', f'{page_height}x{page_width}mm', str(output_path)])
            else:
                cmd.extend(['write', '--page-size', f'{page_width}x{page_height}mm', str(output_path)])
            
            # Execute the command and store subprocess reference
            _current_subprocess = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = _current_subprocess.communicate()
            
            if _current_subprocess.returncode == 0:
                _current_subprocess = None
                return True
            else:
                print(f"Error running vpype: {stderr}")
                _current_subprocess = None
                return False
                
        except FileNotFoundError:
            print("Error: vpype command not found. Please ensure vpype is installed and in your PATH.")
            _current_subprocess = None
            return False
        except Exception as e:
            print(f"Error running vpype: {e}")
            _current_subprocess = None
            return False

    def optimize_svg(self, svg_string, fit_to_margins=0.0, landscape=True, linemerge=True, 
                    tolerance=0.1, linesort=True, reloop=True, linesimplify=True, 
                    page_width=100.0, page_height=148.0):
        """Optimize SVG string using vpype."""
        try:
            # Create temporary files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                input_svg_path = temp_path / "input.svg"
                output_svg_path = temp_path / "output.svg"
                
                # Write input SVG to temporary file
                with open(input_svg_path, 'w', encoding='utf-8') as f:
                    f.write(svg_string)
                
                # Run vpype command
                if not self.run_vpype_command(
                    input_svg_path, output_svg_path, fit_to_margins, landscape, 
                    linemerge, tolerance, linesort, reloop, linesimplify, 
                    page_width, page_height
                ):
                    return ("Error: Failed to run vpype optimization",)
                
                # Read the optimized SVG file and return as string
                if output_svg_path.exists():
                    with open(output_svg_path, 'r', encoding='utf-8') as f:
                        optimized_svg = f.read()
                    return (optimized_svg,)
                else:
                    return ("Error: Optimized SVG file was not created",)
                    
        except Exception as e:
            return (f"Error: {str(e)}",)


class PlotSVG:
    CATEGORY = "Pen Plotter"
    
    @classmethod    
    def INPUT_TYPES(s):
        return { 
            "required": { 
                "svg_string": ("STRING", {"multiline": True}),
            },
            "optional": {
                "layer": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
                "pen_up_speed": ("INT", {"default": 75, "min": 1, "max": 100, "step": 1}),
                "pen_down_speed": ("INT", {"default": 25, "min": 1, "max": 100, "step": 1}),
                "pen_up_delay": ("INT", {"default": 0, "min": 0, "max": 5000, "step": 10}),
                "pen_down_delay": ("INT", {"default": 0, "min": 0, "max": 5000, "step": 10}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "plot_svg"
    OUTPUT_NODE = True

    def run_axicli_plot(self, svg_path, layer, pen_up_speed, pen_down_speed, 
                          pen_up_delay, pen_down_delay):
        """Run axicli command with specified parameters."""
        global _current_subprocess
        try:
            # Build the axicli command
            cmd = ['axicli', str(svg_path)]
            
            # Add layer parameter
            cmd.extend(['-L', str(layer)])
            
            # Add speed settings
            cmd.extend(['-S', str(pen_up_speed)])   # Pen-up speed
            cmd.extend(['-s', str(pen_down_speed)]) # Pen-down speed
            
            # Add delay settings
            if pen_up_delay > 0:
                cmd.extend(['-d', str(pen_up_delay)])   # Pen-up delay
            if pen_down_delay > 0:
                cmd.extend(['-D', str(pen_down_delay)]) # Pen-down delay
            
            # Execute the command and store subprocess reference
            _current_subprocess = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = _current_subprocess.communicate()
            
            if _current_subprocess.returncode == 0:
                _current_subprocess = None
                return True, stdout
            else:
                print(f"Error running axicli: {stderr}")
                _current_subprocess = None
                return False, stderr
                
        except FileNotFoundError:
            error_msg = "Error: axicli command not found. Please ensure AxiDraw software is installed and axicli is in your PATH."
            print(error_msg)
            _current_subprocess = None
            return False, error_msg
        except Exception as e:
            error_msg = f"Error running axicli: {e}"
            print(error_msg)
            _current_subprocess = None
            return False, error_msg

    def plot_svg(self, svg_string, layer=2, pen_up_speed=75, pen_down_speed=25,
                pen_up_delay=0, pen_down_delay=0):
        """Send SVG to plotter using axicli."""
        try:
            # Create temporary file for the SVG
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                svg_path = temp_path / "plot.svg"
                
                # Write SVG to temporary file
                with open(svg_path, 'w', encoding='utf-8') as f:
                    f.write(svg_string)
                
                # Run axicli command
                success, output = self.run_axicli_plot(
                    svg_path, layer, pen_up_speed, pen_down_speed,
                    pen_up_delay, pen_down_delay
                )
                
                if success:
                    run_axicli_disengage()  # Ensure plotter is disengaged after plotting
                    return (f"Plot completed successfully.",)
                else:
                    return (f"Error: {output}",)
                    
        except Exception as e:
            return (f"Error: {str(e)}",)

class DisengagePlotter:
    CATEGORY = "Pen Plotter"
    
    @classmethod    
    def INPUT_TYPES(s):
        return { 
            "required": { 
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "disengage_plotter"
    OUTPUT_NODE = True

    def run_axicli_disengage(self):
        global _current_subprocess
        try:
            # Build the axicli command
            cmd = ['axicli', '--mode', 'align']
            
            # Execute the command and store subprocess reference
            _current_subprocess = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = _current_subprocess.communicate()
            
            if _current_subprocess.returncode == 0:
                _current_subprocess = None
                return True, stdout
            else:
                print(f"Error running axicli: {stderr}")
                _current_subprocess = None
                return False, stderr
                
        except FileNotFoundError:
            error_msg = "Error: axicli command not found. Please ensure AxiDraw software is installed and axicli is in your PATH."
            print(error_msg)
            _current_subprocess = None
            return False, error_msg
        except Exception as e:
            error_msg = f"Error running axicli: {e}"
            print(error_msg)
            _current_subprocess = None
            return False, error_msg

    def disengage_plotter(self):
        """Disengage the plotter using axicli."""
        try:
            success, output = self.run_axicli_disengage()
            if success:
                return ("Plotter disengaged successfully.",)
            else:
                return (f"Error: {output}",)
        except Exception as e:
            return (f"Error: {str(e)}",)

def run_axicli_disengage():
    """Function to disengage the plotter using axicli."""
    try:
        cmd = ['axicli', '--mode', 'align']
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("Plotter disengaged successfully.")
            return True
        else:
            print(f"Error disengaging plotter: {result.stderr}")
            return False
    except FileNotFoundError:
        print("Error: axicli command not found. Please ensure AxiDraw software is installed and axicli is in your PATH.")
        return False
    except Exception as e:
        print(f"Error disengaging plotter: {e}")
        return False

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ImageToCenterline": ImageToCenterline,
    "OptimizeSVG": OptimizeSVG,
    "PlotSVG": PlotSVG,
    "DisengagePlotter": DisengagePlotter
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageToCenterline": "Image to Centerline SVG",
    "OptimizeSVG": "Optimize SVG with vpype",
    "PlotSVG": "Plot SVG with AxiDraw",
    "DisengagePlotter": "Disengage Plotter"
}

# Overwrite or wrap comfy.model_management.interrupt_current_processing
_original_interrupt = comfy.model_management.interrupt_current_processing

def wrapped_interrupt_current_processing(*args, **kwargs):
    global _current_subprocess
    result = _original_interrupt(*args, **kwargs)
    if args and args[0] is True:
        print("[PenPlotter] Processing interrupted.")
        # Interrupt the running subprocess if needed
        if _current_subprocess is not None:
            try:
                print("[PenPlotter] Terminating running subprocess...")
                _current_subprocess.terminate()
                # Give the process a moment to terminate gracefully
                try:
                    _current_subprocess.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    # If it doesn't terminate gracefully, force kill it
                    print("[PenPlotter] Force killing subprocess...")
                    _current_subprocess.kill()
                    _current_subprocess.wait()
                print("[PenPlotter] Subprocess terminated.")
                _current_subprocess = None
            except Exception as e:
                print(f"[PenPlotter] Error terminating subprocess: {e}")
                _current_subprocess = None
        # run the disengage command to put the plotter in a safe state
        run_axicli_disengage()

    return result

comfy.model_management.interrupt_current_processing = wrapped_interrupt_current_processing