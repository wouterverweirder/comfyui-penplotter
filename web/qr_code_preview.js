import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "QRCodePreview";
const PANEL_HEIGHT = 280;
const PANEL_MARGIN = 8;

let animating = false;

function findPreviewNodes() {
    if (!app?.graph?._nodes) return [];
    return app.graph._nodes.filter((n) => n.type === NODE_TYPE);
}

function startAnimating() {
    if (animating) return;
    animating = true;
    const tick = () => {
        let active = false;
        for (const n of findPreviewNodes()) {
            if (n.__qrState === "spinning") {
                active = true;
                n.setDirtyCanvas(true, false);
            }
        }
        if (active) {
            requestAnimationFrame(tick);
        } else {
            animating = false;
        }
    };
    requestAnimationFrame(tick);
}

function imageUrl({ filename, subfolder, type }) {
    const params = new URLSearchParams({
        filename: filename ?? "",
        subfolder: subfolder ?? "",
        type: type ?? "temp",
        rand: String(Date.now()),
    });
    return `/view?${params.toString()}`;
}

function loadImage(node, info) {
    const img = new Image();
    img.onload = () => {
        node.__qrImage = img;
        node.setDirtyCanvas(true, true);
    };
    img.onerror = () => {
        node.__qrImage = null;
        node.__qrState = "error";
        node.setDirtyCanvas(true, true);
    };
    img.src = imageUrl(info);
}

function drawSpinner(ctx, x, y, w, h) {
    const cx = x + w / 2;
    const cy = y + h / 2;
    const r = Math.max(12, Math.min(w, h) * 0.16);
    const t = (Date.now() / 1000) * Math.PI * 1.6;
    const segs = 12;
    for (let i = 0; i < segs; i++) {
        const angle = t + (i / segs) * Math.PI * 2;
        const alpha = (i + 1) / segs;
        ctx.strokeStyle = `rgba(220,220,220,${alpha})`;
        ctx.lineWidth = Math.max(2, r * 0.18);
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(cx + Math.cos(angle) * r * 0.55, cy + Math.sin(angle) * r * 0.55);
        ctx.lineTo(cx + Math.cos(angle) * r, cy + Math.sin(angle) * r);
        ctx.stroke();
    }
    ctx.fillStyle = "rgba(220,220,220,0.85)";
    ctx.font = "12px Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText("Uploading…", cx, cy + r + 10);
}

function drawError(ctx, x, y, w, h) {
    const cx = x + w / 2;
    const cy = y + h / 2;
    const r = Math.max(14, Math.min(w, h) * 0.16);

    ctx.strokeStyle = "#ff5757";
    ctx.fillStyle = "#ff5757";
    ctx.lineWidth = Math.max(3, r * 0.18);
    ctx.lineCap = "round";

    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(cx, cy - r * 0.45);
    ctx.lineTo(cx, cy + r * 0.05);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(cx, cy + r * 0.32, Math.max(2, r * 0.1), 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "rgba(255,180,180,0.95)";
    ctx.font = "12px Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText("Upload failed", cx, cy + r + 10);
}

function drawImageFitted(ctx, img, x, y, w, h) {
    const ar = img.width / img.height;
    const boxAr = w / h;
    let dw, dh;
    if (ar > boxAr) {
        dw = w;
        dh = w / ar;
    } else {
        dh = h;
        dw = h * ar;
    }
    const dx = x + (w - dw) / 2;
    const dy = y + (h - dh) / 2;
    ctx.drawImage(img, dx, dy, dw, dh);
}

app.registerExtension({
    name: "comfyui-penplotter.QRCodePreview",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            this.__qrState = "idle";
            this.__qrImage = null;

            const widget = {
                type: "qr_panel",
                name: "qr_panel",
                draw(ctx, node, widgetWidth, posY) {
                    const x = PANEL_MARGIN;
                    const y = posY;
                    const w = Math.max(40, widgetWidth - PANEL_MARGIN * 2);
                    const h = PANEL_HEIGHT;

                    ctx.fillStyle = "#1a1a1a";
                    ctx.fillRect(x, y, w, h);
                    ctx.strokeStyle = "#2a2a2a";
                    ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);

                    if (node.__qrState === "spinning") {
                        drawSpinner(ctx, x, y, w, h);
                    } else if (node.__qrState === "error") {
                        drawError(ctx, x, y, w, h);
                    } else if (node.__qrImage) {
                        drawImageFitted(ctx, node.__qrImage, x + 4, y + 4, w - 8, h - 8);
                    }
                },
                computeSize(width) {
                    return [width, PANEL_HEIGHT + PANEL_MARGIN];
                },
                serializeValue() {
                    return null;
                },
            };
            this.addCustomWidget(widget);
            this.setSize(this.computeSize());
            return r;
        };
    },

    setup() {
        api.addEventListener("execution_start", () => {
            for (const n of findPreviewNodes()) {
                n.__qrState = "spinning";
                n.__qrImage = null;
                n.setDirtyCanvas(true, true);
            }
            startAnimating();
        });

        // Fired by UploadSubmission as soon as the API call succeeds, before
        // any long-running downstream node (e.g. PlotSVG) starts. This is the
        // primary path that surfaces the QR quickly.
        api.addEventListener("penplotter.qr_ready", ({ detail }) => {
            const filename = detail?.filename;
            if (!filename) return;
            const info = {
                filename,
                subfolder: detail?.subfolder ?? "",
                type: detail?.type ?? "temp",
            };
            for (const n of findPreviewNodes()) {
                n.__qrState = "ready";
                loadImage(n, info);
            }
        });

        // Fallback: fires when QRCodePreview itself executes. May arrive
        // after a long downstream node finishes, so it's only useful if the
        // early push above didn't run (e.g. QRCodePreview without
        // UploadSubmission upstream).
        api.addEventListener("executed", ({ detail }) => {
            const id = detail?.node;
            if (id == null) return;
            const node = app.graph.getNodeById(id);
            if (!node || node.type !== NODE_TYPE) return;
            if (node.__qrState === "ready" && node.__qrImage) return;
            const images = detail?.output?.qr_images;
            if (Array.isArray(images) && images.length > 0) {
                node.__qrState = "ready";
                loadImage(node, images[0]);
            } else {
                node.__qrState = "error";
                node.setDirtyCanvas(true, true);
            }
        });

        const fail = () => {
            for (const n of findPreviewNodes()) {
                if (n.__qrState === "spinning") {
                    n.__qrState = "error";
                    n.setDirtyCanvas(true, true);
                }
            }
        };
        api.addEventListener("execution_error", fail);
        api.addEventListener("execution_interrupted", fail);
    },
});
