import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { Muxer, ArrayBufferTarget } from "./mp4-muxer.mjs";

const CODEC = "avc1.42E01F";
const BITRATE = 4_000_000;

function parseSvg(svgString) {
    const doc = new DOMParser().parseFromString(svgString, "image/svg+xml");
    if (doc.querySelector("parsererror")) {
        throw new Error("SVG parse error");
    }
    return doc.documentElement;
}

function getSvgDimensions(svgEl) {
    const viewBox = (svgEl.getAttribute("viewBox") || "").split(/\s+/).map(Number);
    if (viewBox.length === 4 && viewBox.every((n) => Number.isFinite(n))) {
        return { width: viewBox[2], height: viewBox[3] };
    }
    const w = parseFloat(svgEl.getAttribute("width"));
    const h = parseFloat(svgEl.getAttribute("height"));
    if (Number.isFinite(w) && Number.isFinite(h)) {
        return { width: w, height: h };
    }
    return { width: 537, height: 374 };
}

function collectShapes(svgEl) {
    return Array.from(svgEl.querySelectorAll("polyline, line, path"));
}

function computeEffectiveSpeed(shapes, speedPxPerSec, maxDurationSeconds) {
    const totalLength = shapes.reduce((sum, el) => sum + el.getTotalLength(), 0);
    const minSpeed = totalLength / Math.max(0.001, maxDurationSeconds);
    return Math.max(speedPxPerSec, minSpeed);
}

function buildTimeline(shapes, speed) {
    let cumulative = 0;
    const items = shapes.map((el) => {
        const len = el.getTotalLength();
        const item = { el, len, start: cumulative };
        cumulative += len / speed;
        return item;
    });
    return { items, totalDuration: cumulative };
}

function applyDashoffsetForTime(items, t, speed) {
    for (const { el, len, start } of items) {
        const finishTime = start + len / speed;
        let offset;
        if (t <= start) {
            offset = len;
        } else if (t >= finishTime) {
            offset = 0;
        } else {
            offset = len - (t - start) * speed;
        }
        el.setAttribute("stroke-dasharray", len);
        el.setAttribute("stroke-dashoffset", offset);
    }
}

// SVG must be attached to a document for getTotalLength() and the Image-based
// rasterization to work reliably.
function getOffscreenHost() {
    let host = document.getElementById("__penplotter_offscreen");
    if (!host) {
        host = document.createElement("div");
        host.id = "__penplotter_offscreen";
        host.setAttribute("aria-hidden", "true");
        host.style.position = "absolute";
        host.style.left = "-9999px";
        host.style.top = "0";
        host.style.visibility = "hidden";
        host.style.pointerEvents = "none";
        document.body.appendChild(host);
    }
    return host;
}

async function renderSvgToMp4(svgString, settings) {
    if (!("VideoEncoder" in window)) {
        throw new Error("VideoEncoder is not supported in this browser. Use Chrome/Edge/Safari 16.4+.");
    }

    const speed_px_per_sec = Number(settings.speed_px_per_sec) || 400;
    const max_duration_seconds = Number(settings.max_duration_seconds) || 30;
    const fps = Math.max(1, Number(settings.fps) || 30);
    const out_w = Math.max(2, Math.round(Number(settings.out_width) || 1280));
    const out_h = Math.max(2, Math.round(Number(settings.out_height) || 720));
    const bg_color = String(settings.bg_color || "#fafafa");
    const freeze_final_seconds = Math.max(0, Number(settings.freeze_final_seconds) || 0);
    const stroke_width = Math.max(0, Number(settings.stroke_width) || 0);

    const support = await VideoEncoder.isConfigSupported({
        codec: CODEC, width: out_w, height: out_h, bitrate: BITRATE, framerate: fps,
    }).catch(() => null);
    if (!support || !support.supported) {
        throw new Error(`H.264 (${CODEC}) at ${out_w}x${out_h}@${fps} not supported by VideoEncoder in this browser.`);
    }

    const svgEl = parseSvg(svgString);
    const { width: srcW, height: srcH } = getSvgDimensions(svgEl);
    const scale = Math.min(out_w / srcW, out_h / srcH);
    const dw = Math.round(srcW * scale);
    const dh = Math.round(srcH * scale);
    const dx = Math.round((out_w - dw) / 2);
    const dy = Math.round((out_h - dh) / 2);

    svgEl.setAttribute("width", dw);
    svgEl.setAttribute("height", dh);
    const host = getOffscreenHost();
    host.appendChild(svgEl);

    try {
        const shapes = collectShapes(svgEl);
        if (shapes.length === 0) {
            throw new Error("SVG has no <polyline>, <line>, or <path> elements to animate.");
        }
        if (stroke_width > 0) {
            for (const el of shapes) {
                el.setAttribute("stroke-width", stroke_width);
            }
        }
        const speed = computeEffectiveSpeed(shapes, speed_px_per_sec, max_duration_seconds);
        const { items, totalDuration } = buildTimeline(shapes, speed);
        applyDashoffsetForTime(items, 0, speed);

        const drawingFrames = Math.max(1, Math.ceil(totalDuration * fps) + 1);
        const freezeFrames = Math.max(0, Math.round(freeze_final_seconds * fps));

        const canvas = ("OffscreenCanvas" in window)
            ? new OffscreenCanvas(out_w, out_h)
            : Object.assign(document.createElement("canvas"), { width: out_w, height: out_h });
        const ctx = canvas.getContext("2d");

        const muxer = new Muxer({
            target: new ArrayBufferTarget(),
            video: { codec: "avc", width: out_w, height: out_h },
            fastStart: "in-memory",
        });

        // Safari/older Chrome occasionally hand back a chunk meta without a
        // colorSpace, which mp4-muxer refuses. Cache the first valid config
        // and reuse it for every subsequent chunk.
        const fallbackColorSpace = {
            primaries: "bt709", transfer: "bt709", matrix: "bt709", fullRange: false,
        };
        const frameDurationUs = Math.floor(1_000_000 / fps);
        let cachedDecoderConfig = null;

        const encoder = new VideoEncoder({
            output(chunk, meta) {
                if (meta?.decoderConfig) {
                    cachedDecoderConfig = {
                        ...meta.decoderConfig,
                        colorSpace: meta.decoderConfig.colorSpace ?? fallbackColorSpace,
                    };
                }
                const safeMeta = cachedDecoderConfig ? { decoderConfig: cachedDecoderConfig } : meta;
                const data = new Uint8Array(chunk.byteLength);
                chunk.copyTo(data);
                const duration = chunk.duration ?? frameDurationUs;
                muxer.addVideoChunkRaw(data, chunk.type, chunk.timestamp, duration, safeMeta);
            },
            error(e) {
                console.error("[penplotter] VideoEncoder error:", e);
            },
        });

        encoder.configure({
            codec: CODEC, width: out_w, height: out_h,
            bitrate: BITRATE, framerate: fps,
            avc: { format: "avc" },
        });

        const serializer = new XMLSerializer();
        const wallStart = performance.now();

        for (let i = 0; i < drawingFrames; i++) {
            const t = i / fps;
            applyDashoffsetForTime(items, t, speed);

            const svgStr = serializer.serializeToString(svgEl);
            const blob = new Blob([svgStr], { type: "image/svg+xml" });
            const blobUrl = URL.createObjectURL(blob);
            const img = new Image();
            img.src = blobUrl;
            try {
                await img.decode();
            } catch (err) {
                URL.revokeObjectURL(blobUrl);
                throw new Error(`Failed to decode SVG frame ${i}: ${err.message}`);
            }

            ctx.fillStyle = bg_color;
            ctx.fillRect(0, 0, out_w, out_h);
            ctx.drawImage(img, dx, dy, dw, dh);
            URL.revokeObjectURL(blobUrl);

            const frame = new VideoFrame(canvas, {
                timestamp: Math.floor(i * 1_000_000 / fps),
                duration: frameDurationUs,
            });
            encoder.encode(frame, { keyFrame: (i % fps) === 0 });
            frame.close();

            while (encoder.encodeQueueSize > 4) {
                await new Promise((r) => setTimeout(r, 0));
            }
        }

        for (let j = 0; j < freezeFrames; j++) {
            const i = drawingFrames + j;
            const frame = new VideoFrame(canvas, {
                timestamp: Math.floor(i * 1_000_000 / fps),
                duration: frameDurationUs,
            });
            encoder.encode(frame, { keyFrame: (i % fps) === 0 });
            frame.close();

            while (encoder.encodeQueueSize > 4) {
                await new Promise((r) => setTimeout(r, 0));
            }
        }

        await encoder.flush();
        muxer.finalize();
        encoder.close();

        const wallTotal = (performance.now() - wallStart) / 1000;
        const totalFrames = drawingFrames + freezeFrames;
        console.log(
            `[penplotter] encoded ${totalFrames} frames (${(totalFrames / fps).toFixed(1)}s video) in ${wallTotal.toFixed(1)}s`,
        );

        return new Blob([muxer.target.buffer], { type: "video/mp4" });
    } finally {
        if (svgEl.parentNode === host) {
            host.removeChild(svgEl);
        }
    }
}

app.registerExtension({
    name: "comfyui-penplotter.SvgToMp4Animation",
    setup() {
        api.addEventListener("penplotter.render_mp4", async (event) => {
            const detail = event.detail || {};
            const { job_id, svg, settings } = detail;
            if (!job_id || typeof svg !== "string") {
                console.error("[penplotter] render_mp4 event missing job_id or svg", detail);
                return;
            }
            try {
                console.log(`[penplotter] rendering mp4 for job ${job_id}`, settings);
                const mp4Blob = await renderSvgToMp4(svg, settings || {});

                const fd = new FormData();
                fd.append("job_id", job_id);
                fd.append("file", mp4Blob, `${job_id}.mp4`);
                const resp = await api.fetchApi("/penplotter/mp4_result", { method: "POST", body: fd });
                if (!resp.ok) {
                    throw new Error(`upload failed: HTTP ${resp.status}`);
                }
                console.log(`[penplotter] uploaded mp4 for job ${job_id} (${(mp4Blob.size / 1024).toFixed(0)} KB)`);
            } catch (err) {
                console.error("[penplotter] mp4 render failed", err);
                try {
                    await api.fetchApi("/penplotter/mp4_error", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ job_id, error: String((err && err.message) || err) }),
                    });
                } catch (innerErr) {
                    console.error("[penplotter] failed to report error to backend", innerErr);
                }
            }
        });
    },
});
