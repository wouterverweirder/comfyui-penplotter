from inspect import cleandoc
import numpy as np
import tempfile
import subprocess
import os
import json
import random
import string
import uuid
import threading
from io import BytesIO
from pathlib import Path
from aiohttp import web
from server import PromptServer
import cv2
import torch
import requests
import qrcode
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths
from .trace_skeleton import thinning, traceSkeleton
import comfy.model_management
from comfy_api.input_impl import VideoFromFile


# Global variable to track the currently running subprocess for interruption
_current_subprocess = None

# Pending browser-side MP4 render jobs keyed by job_id.
# Each entry: {"event": threading.Event, "result": str|None, "error": str|None}
_mp4_jobs = {}
_mp4_jobs_lock = threading.Lock()

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
                "color": ("STRING", {"default": "black"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("svg_string",)
    FUNCTION = "convert_to_centerline"

    def tensor_to_cv2(self, tensor):
        """Convert ComfyUI tensor to OpenCV image."""
        # Convert tensor to numpy array
        # ComfyUI tensors are in format (batch, height, width, channels)
        np_image = tensor.cpu().numpy()
        
        # Get first image from batch if batched
        if len(np_image.shape) == 4:
            np_image = np_image[0]
        
        # Convert from float [0,1] to uint8 [0,255]
        np_image = (np_image * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        return cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

    def convert_to_centerline(self, image, optimize_exhaustive=True, color="black"):
        """Convert image tensor to centerline SVG string."""

        try:

            # create a cv2 image from PIL image
            im0 = self.tensor_to_cv2(image)

            # invert colors
            im0 = 255 - im0

            im = (im0[:,:,0]>128).astype(np.uint8)
            im = thinning(im)
            rects = []
            polys = traceSkeleton(im,0,0,im.shape[1],im.shape[0],10,999,rects)

            return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{im.shape[1]}" height="{im.shape[0]}"><path stroke="{color}" fill="none" d="'+" ".join(
    ["M"+" ".join([f'{x[0]},{x[1]}' for x in y]) for y in polys]
  )+'"/></svg>',)
                    
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
                "wait_for": ("STRING", {"forceInput": True}),
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
                pen_up_delay=0, pen_down_delay=0, wait_for=None):
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

class ImageToFile:
    """Wrap a ComfyUI IMAGE tensor as a PENPLOTTER_FILE for upload."""
    CATEGORY = "Pen Plotter"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "name": ("STRING", {"default": "image.png"}),
            }
        }

    RETURN_TYPES = ("PENPLOTTER_FILE",)
    RETURN_NAMES = ("file",)
    FUNCTION = "to_file"

    def to_file(self, image, name):
        np_image = image.cpu().numpy()
        if np_image.ndim == 4:
            np_image = np_image[0]
        np_image = np.clip(np_image * 255.0, 0, 255).astype(np.uint8)
        pil = Image.fromarray(np_image)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")

        buf = BytesIO()
        pil.save(buf, format="PNG")

        cleaned = (name or "image").strip() or "image"
        if not os.path.splitext(cleaned)[1]:
            cleaned = f"{cleaned}.png"

        return ({
            "name": cleaned,
            "bytes": buf.getvalue(),
            "mime": "image/png",
        },)


class SvgToFile:
    """Wrap an SVG string as a PENPLOTTER_FILE for upload."""
    CATEGORY = "Pen Plotter"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "svg_string": ("STRING", {"multiline": True, "forceInput": True}),
                "name": ("STRING", {"default": "drawing.svg"}),
            }
        }

    RETURN_TYPES = ("PENPLOTTER_FILE",)
    RETURN_NAMES = ("file",)
    FUNCTION = "to_file"

    def to_file(self, svg_string, name):
        cleaned = (name or "drawing").strip() or "drawing"
        if not os.path.splitext(cleaned)[1]:
            cleaned = f"{cleaned}.svg"

        return ({
            "name": cleaned,
            "bytes": (svg_string or "").encode("utf-8"),
            "mime": "image/svg+xml",
        },)


class VideoToFile:
    """Wrap a VIDEO input as a PENPLOTTER_FILE for UploadSubmission."""
    CATEGORY = "Pen Plotter"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("VIDEO",),
                "name": ("STRING", {"default": "animation.mp4"}),
            }
        }

    RETURN_TYPES = ("PENPLOTTER_FILE",)
    RETURN_NAMES = ("file",)
    FUNCTION = "to_file"

    def to_file(self, video, name):
        try:
            src = video.get_stream_source()
            if isinstance(src, str):
                with open(src, "rb") as f:
                    data = f.read()
            else:
                src.seek(0)
                data = src.read()
        except Exception:
            # VideoFromComponents (e.g. from core CreateVideo) doesn't override
            # get_stream_source; the base default calls save_to(BytesIO()) with
            # format=AUTO, which PyAV can't resolve. A path with a .mp4
            # extension lets PyAV infer the container.
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = os.path.join(tmp_dir, "out.mp4")
                video.save_to(tmp_path)
                with open(tmp_path, "rb") as f:
                    data = f.read()

        cleaned = (name or "animation").strip() or "animation"
        if not os.path.splitext(cleaned)[1]:
            cleaned = f"{cleaned}.mp4"

        return ({
            "name": cleaned,
            "bytes": data,
            "mime": "video/mp4",
        },)


class UploadSubmission:
    """POST collected files to the AI Plotter Laravel backend and render the
    resulting public URL as a QR code image. The `project` widget is a label
    for the operator's reference only — the project is derived server-side
    from the API key."""
    CATEGORY = "Pen Plotter"

    _ERROR_CORRECTION = {
        "L": qrcode.constants.ERROR_CORRECT_L,
        "M": qrcode.constants.ERROR_CORRECT_M,
        "Q": qrcode.constants.ERROR_CORRECT_Q,
        "H": qrcode.constants.ERROR_CORRECT_H,
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_endpoint": ("STRING", {"default": "https://aiplotter.devine.be"}),
                "project": ("STRING", {"default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "box_size": ("INT", {"default": 10, "min": 1, "max": 50, "step": 1}),
                "error_correction": (["L", "M", "Q", "H"], {"default": "M"}),
            },
            "optional": {
                "file_1": ("PENPLOTTER_FILE",),
                "file_2": ("PENPLOTTER_FILE",),
                "file_3": ("PENPLOTTER_FILE",),
                "file_4": ("PENPLOTTER_FILE",),
                "file_5": ("PENPLOTTER_FILE",),
                "file_6": ("PENPLOTTER_FILE",),
                "file_7": ("PENPLOTTER_FILE",),
                "file_8": ("PENPLOTTER_FILE",),
                "file_9": ("PENPLOTTER_FILE",),
                "file_10": ("PENPLOTTER_FILE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("qr_code", "public_url", "token")
    FUNCTION = "upload"

    def upload(self, api_endpoint, project, api_key, box_size, error_correction,
               file_1=None, file_2=None, file_3=None, file_4=None, file_5=None,
               file_6=None, file_7=None, file_8=None, file_9=None, file_10=None):
        slots = [file_1, file_2, file_3, file_4, file_5,
                 file_6, file_7, file_8, file_9, file_10]
        files = [f for f in slots if f is not None]
        if not files:
            raise RuntimeError("No files attached to UploadSubmission — wire at least one PENPLOTTER_FILE input.")

        if not api_key.strip():
            raise RuntimeError("UploadSubmission: api_key is empty.")
        if not api_endpoint.strip():
            raise RuntimeError("UploadSubmission: api_endpoint is empty.")

        url = f"{api_endpoint.rstrip('/')}/api/v1/submissions"
        files_payload = []
        data_payload = {}
        for i, f in enumerate(files):
            files_payload.append((f"files[{i}][file]", (f["name"], f["bytes"], f["mime"])))
            data_payload[f"files[{i}][label]"] = f["name"]

        print(f"[PenPlotter] Uploading {len(files)} file(s) to {url}")
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key.strip()}",
                "Accept": "application/json",
            },
            files=files_payload,
            data=data_payload,
            timeout=60,
        )

        if not response.ok:
            body = response.text or ""
            if len(body) > 800:
                body = body[:800] + "…"
            raise RuntimeError(f"UploadSubmission failed: HTTP {response.status_code} — {body}")

        try:
            payload = response.json()
        except ValueError as e:
            raise RuntimeError(f"UploadSubmission: response was not valid JSON — {response.text[:400]}") from e

        public_url = payload.get("public_url", "")
        token = payload.get("token", "")
        if not public_url:
            raise RuntimeError(f"UploadSubmission: response missing public_url — {payload}")

        qr = qrcode.QRCode(
            version=None,
            error_correction=self._ERROR_CORRECTION[error_correction],
            box_size=int(box_size),
            border=4,
        )
        qr.add_data(public_url)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")

        # Push the QR to the frontend immediately, before downstream nodes
        # (PlotSVG can take many minutes — we don't want the user staring at
        # a spinner that long). QRCodePreview's own executed event still
        # fires later but is redundant by then.
        try:
            temp_dir = folder_paths.get_temp_directory()
            push_prefix = "qr_push_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
            push_folder, push_filename, push_counter, push_subfolder, _ = folder_paths.get_save_image_path(
                push_prefix, temp_dir, qr_img.size[0], qr_img.size[1]
            )
            push_file = f"{push_filename}_{push_counter:05}_.png"
            qr_img.save(os.path.join(push_folder, push_file), compress_level=1)
            PromptServer.instance.send_sync("penplotter.qr_ready", {
                "filename": push_file,
                "subfolder": push_subfolder,
                "type": "temp",
                "public_url": public_url,
                "token": token,
            })
        except Exception as e:
            print(f"[PenPlotter] Could not push early QR update: {e}")

        qr_np = np.array(qr_img).astype(np.float32) / 255.0
        qr_tensor = torch.from_numpy(qr_np).unsqueeze(0)

        print(f"[PenPlotter] Submission token={token} public_url={public_url}")
        return (qr_tensor, public_url, token)


class QRCodePreview:
    """Sink node for the QR code image. Pairs with the custom frontend
    extension in `web/qr_code_preview.js` which shows a spinner during the
    workflow run and an error icon on failure/interrupt. Uses a custom
    `qr_images` ui key so ComfyUI's default thumbnail strip doesn't also
    render the image."""
    CATEGORY = "Pen Plotter"

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_qr_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    def save(self, image, prompt=None, extra_pnginfo=None):
        first = image[0]
        height, width = int(first.shape[0]), int(first.shape[1])
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            "qr_preview" + self.prefix_append, self.output_dir, width, height
        )

        results = []
        for img_tensor in image:
            np_img = np.clip(img_tensor.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            pil = Image.fromarray(np_img)
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for k, v in extra_pnginfo.items():
                    metadata.add_text(k, json.dumps(v))
            file_name = f"{filename}_{counter:05}_.png"
            pil.save(os.path.join(full_output_folder, file_name),
                     pnginfo=metadata, compress_level=self.compress_level)
            results.append({"filename": file_name, "subfolder": subfolder, "type": self.type})
            counter += 1

        return {"ui": {"qr_images": results}}


class VideoPreview:
    """Sink node that plays a VIDEO inline in the graph. Pairs with
    web/video_preview.js, which mounts a <video> element with autoplay+loop+
    muted and reloads whenever a new render finishes."""
    CATEGORY = "Pen Plotter"

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_vid_" + "".join(
            random.choices(string.ascii_lowercase + string.digits, k=6)
        )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"video": ("VIDEO",)},
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True

    def preview(self, video, unique_id=None):
        os.makedirs(self.output_dir, exist_ok=True)
        try:
            width, height = video.get_dimensions()
        except Exception:
            width, height = 0, 0

        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            "video_preview" + self.prefix_append,
            self.output_dir, width, height,
        )
        file_name = f"{filename}_{counter:05}_.mp4"
        out_path = os.path.join(full_output_folder, file_name)
        video.save_to(out_path)

        result = {"filename": file_name, "subfolder": subfolder, "type": self.type}

        # Early push so the frontend can swap the <video src> the moment the
        # file is on disk (mirrors the qr_ready pattern used by UploadSubmission).
        try:
            PromptServer.instance.send_sync(
                "penplotter.video_ready",
                {**result, "node_type": "VideoPreview", "node_id": unique_id},
            )
        except Exception as e:
            print(f"[PenPlotter] Could not push video_ready: {e}")

        return {"ui": {"penplotter_videos": [result]}}


class SvgToMp4Animation:
    """Render an SVG line-drawing as an MP4 by delegating encoding to the
    browser (WebCodecs VideoEncoder + mp4-muxer). The Python node blocks on
    a per-job threading.Event while the frontend extension does the work
    and POSTs the finished MP4 back to /penplotter/mp4_result."""
    CATEGORY = "Pen Plotter"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "svg": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "speed_px_per_sec":     ("INT",   {"default": 400,   "min": 1,    "max": 10000}),
                "max_duration_seconds": ("FLOAT", {"default": 30.0,  "min": 1.0,  "max": 600.0, "step": 0.5}),
                "fps":                  ("INT",   {"default": 30,    "min": 1,    "max": 120}),
                "out_width":            ("INT",   {"default": 1280,  "min": 16,   "max": 7680}),
                "out_height":           ("INT",   {"default": 720,   "min": 16,   "max": 4320}),
                "bg_color":             ("STRING",{"default": "#fafafa"}),
                "freeze_final_seconds": ("FLOAT", {"default": 1.0,   "min": 0.0,  "max": 60.0, "step": 0.1}),
                "stroke_width":         ("FLOAT", {"default": 1.0,   "min": 0.0,  "max": 100.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "render"

    def render(self, svg, speed_px_per_sec=400, max_duration_seconds=30.0, fps=30,
               out_width=1280, out_height=720, bg_color="#fafafa", freeze_final_seconds=1.0,
               stroke_width=1.0):
        job_id = uuid.uuid4().hex
        event = threading.Event()
        with _mp4_jobs_lock:
            _mp4_jobs[job_id] = {"event": event, "result": None, "error": None}

        try:
            PromptServer.instance.send_sync("penplotter.render_mp4", {
                "job_id": job_id,
                "svg": svg,
                "settings": {
                    "speed_px_per_sec": int(speed_px_per_sec),
                    "max_duration_seconds": float(max_duration_seconds),
                    "fps": int(fps),
                    "out_width": int(out_width),
                    "out_height": int(out_height),
                    "bg_color": bg_color,
                    "freeze_final_seconds": float(freeze_final_seconds),
                    "stroke_width": float(stroke_width),
                },
            })

            # Generous timeout: full animation length + freeze + 2 min slack
            # for encoder warm-up and upload of larger files.
            timeout = float(max_duration_seconds) + float(freeze_final_seconds) + 120.0
            if not event.wait(timeout=timeout):
                raise RuntimeError(
                    "Timed out waiting for browser to render MP4. "
                    "Is the ComfyUI tab open?"
                )

            with _mp4_jobs_lock:
                job = _mp4_jobs.get(job_id)
            if job is None:
                raise RuntimeError("MP4 job state was lost.")
            if job["error"]:
                raise RuntimeError(f"Browser-side MP4 render failed: {job['error']}")
            if not job["result"]:
                raise RuntimeError("Browser returned no MP4 file.")

            return (VideoFromFile(job["result"]),)
        finally:
            with _mp4_jobs_lock:
                _mp4_jobs.pop(job_id, None)


@PromptServer.instance.routes.post("/penplotter/mp4_result")
async def _penplotter_receive_mp4(request):
    reader = await request.multipart()
    job_id = None
    file_bytes = None
    while True:
        field = await reader.next()
        if field is None:
            break
        if field.name == "job_id":
            job_id = (await field.read()).decode("utf-8").strip()
        elif field.name == "file":
            file_bytes = await field.read()
    if not job_id or file_bytes is None:
        return web.json_response({"error": "missing job_id or file"}, status=400)

    temp_dir = folder_paths.get_temp_directory()
    os.makedirs(temp_dir, exist_ok=True)
    out_path = os.path.join(temp_dir, f"penplotter_anim_{job_id}.mp4")
    with open(out_path, "wb") as f:
        f.write(file_bytes)

    with _mp4_jobs_lock:
        job = _mp4_jobs.get(job_id)
        if job is not None:
            job["result"] = out_path
            job["event"].set()
    return web.json_response({"ok": True})


@PromptServer.instance.routes.post("/penplotter/mp4_error")
async def _penplotter_receive_mp4_error(request):
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    job_id = (payload or {}).get("job_id")
    err = (payload or {}).get("error", "unknown error")
    if not job_id:
        return web.json_response({"error": "missing job_id"}, status=400)
    with _mp4_jobs_lock:
        job = _mp4_jobs.get(job_id)
        if job is not None:
            job["error"] = str(err)
            job["event"].set()
    return web.json_response({"ok": True})


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ImageToCenterline": ImageToCenterline,
    "OptimizeSVG": OptimizeSVG,
    "PlotSVG": PlotSVG,
    "DisengagePlotter": DisengagePlotter,
    "ImageToFile": ImageToFile,
    "SvgToFile": SvgToFile,
    "VideoToFile": VideoToFile,
    "UploadSubmission": UploadSubmission,
    "QRCodePreview": QRCodePreview,
    "VideoPreview": VideoPreview,
    "SvgToMp4Animation": SvgToMp4Animation,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageToCenterline": "Image to Centerline SVG",
    "OptimizeSVG": "Optimize SVG with vpype",
    "PlotSVG": "Plot SVG with AxiDraw",
    "DisengagePlotter": "Disengage Plotter",
    "ImageToFile": "Image to Submission File",
    "SvgToFile": "SVG to Submission File",
    "VideoToFile": "Video to Submission File",
    "UploadSubmission": "Upload Submission to AI Plotter",
    "QRCodePreview": "QR Code Preview",
    "VideoPreview": "Video Preview",
    "SvgToMp4Animation": "SVG to MP4 Animation",
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
        # Wake up any browser-side MP4 jobs that are still blocking the
        # worker so a Cancel returns promptly instead of waiting for the
        # full job timeout.
        with _mp4_jobs_lock:
            for job in _mp4_jobs.values():
                if not job["event"].is_set():
                    job["error"] = "interrupted"
                    job["event"].set()
        # run the disengage command to put the plotter in a safe state
        run_axicli_disengage()

    return result

comfy.model_management.interrupt_current_processing = wrapped_interrupt_current_processing