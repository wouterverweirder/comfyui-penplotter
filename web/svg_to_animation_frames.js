import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

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

// Offscreen DOM container — the SVG must be attached to a document for
// getTotalLength() and the Image-based rasterization to work reliably.
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

async function canvasToPngBlob(canvas) {
    if (typeof canvas.convertToBlob === "function") {
        return await canvas.convertToBlob({ type: "image/png" });
    }
    return await new Promise((resolve, reject) => {
        canvas.toBlob((blob) => {
            if (blob) resolve(blob);
            else reject(new Error("canvas.toBlob returned null"));
        }, "image/png");
    });
}

async function renderSvgToFrames(svgString, settings) {
    const speed_px_per_sec = Number(settings.speed_px_per_sec) || 400;
    const max_duration_seconds = Number(settings.max_duration_seconds) || 30;
    const fps = Math.max(1, Number(settings.fps) || 30);
    const out_w = Math.max(2, Math.round(Number(settings.out_width) || 1280));
    const out_h = Math.max(2, Math.round(Number(settings.out_height) || 720));
    const bg_color = String(settings.bg_color || "#fafafa");
    const freeze_final_seconds = Math.max(0, Number(settings.freeze_final_seconds) || 0);

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
        const speed = computeEffectiveSpeed(shapes, speed_px_per_sec, max_duration_seconds);
        const { items, totalDuration } = buildTimeline(shapes, speed);
        applyDashoffsetForTime(items, 0, speed);

        const drawingFrames = Math.max(1, Math.ceil(totalDuration * fps) + 1);
        const freezeFrames = Math.max(0, Math.round(freeze_final_seconds * fps));

        const canvas = ("OffscreenCanvas" in window)
            ? new OffscreenCanvas(out_w, out_h)
            : Object.assign(document.createElement("canvas"), { width: out_w, height: out_h });
        const ctx = canvas.getContext("2d");

        const serializer = new XMLSerializer();
        const frameBlobs = [];

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

            frameBlobs.push(await canvasToPngBlob(canvas));

            // Yield to the event loop periodically so the UI stays responsive.
            if ((i & 7) === 0) {
                await new Promise((r) => setTimeout(r, 0));
            }
        }

        return {
            frameBlobs,
            freezeFrames,
            width: out_w,
            height: out_h,
            fps,
        };
    } finally {
        if (svgEl.parentNode === host) {
            host.removeChild(svgEl);
        }
    }
}

app.registerExtension({
    name: "penplotter.SvgToAnimationFrames",
    setup() {
        api.addEventListener("penplotter.render_frames", async (event) => {
            const detail = event.detail || {};
            const { job_id, svg, settings } = detail;
            if (!job_id || typeof svg !== "string") {
                console.error("[penplotter] render_frames event missing job_id or svg", detail);
                return;
            }
            try {
                console.log(`[penplotter] rendering frames for job ${job_id}`, settings);
                const result = await renderSvgToFrames(svg, settings || {});

                const fd = new FormData();
                fd.append("job_id", job_id);
                fd.append("fps", String(result.fps));
                fd.append("width", String(result.width));
                fd.append("height", String(result.height));
                fd.append("drawing_frame_count", String(result.frameBlobs.length));
                fd.append("freeze_frame_count", String(result.freezeFrames));
                for (let i = 0; i < result.frameBlobs.length; i++) {
                    fd.append(`frame_${i}`, result.frameBlobs[i], `frame_${i}.png`);
                }

                const resp = await api.fetchApi("/penplotter/frames_result", { method: "POST", body: fd });
                if (!resp.ok) {
                    throw new Error(`upload failed: HTTP ${resp.status}`);
                }
                console.log(`[penplotter] uploaded ${result.frameBlobs.length} frame(s) for job ${job_id} (+${result.freezeFrames} freeze)`);
            } catch (err) {
                console.error("[penplotter] frame render failed", err);
                try {
                    await api.fetchApi("/penplotter/frames_error", {
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
