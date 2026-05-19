import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "VideoPreview";
const PANEL_HEIGHT = 280;

function findPreviewNodes() {
    if (!app?.graph?._nodes) return [];
    return app.graph._nodes.filter((n) => n.type === NODE_TYPE);
}

function videoUrl({ filename, subfolder, type }) {
    const params = new URLSearchParams({
        filename: filename ?? "",
        subfolder: subfolder ?? "",
        type: type ?? "temp",
        rand: String(Date.now()),
    });
    return `/view?${params.toString()}`;
}

function setSource(node, info) {
    const el = node.__videoEl;
    if (!el) return;
    el.src = videoUrl(info);
    el.load();
    el.play().catch(() => { /* autoplay may be blocked; user can click play */ });
}

app.registerExtension({
    name: "comfyui-penplotter.VideoPreview",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);

            const el = document.createElement("video");
            el.autoplay = true;
            el.loop = true;
            el.muted = true;
            el.playsInline = true;
            el.controls = true;
            el.style.width = "100%";
            el.style.height = "100%";
            el.style.background = "#1a1a1a";
            el.style.objectFit = "contain";

            this.__videoEl = el;
            this.addDOMWidget("video_panel", "video", el, {
                serialize: false,
                getHeight: () => PANEL_HEIGHT,
            });
            this.setSize(this.computeSize());
            return r;
        };
    },

    setup() {
        api.addEventListener("penplotter.video_ready", ({ detail }) => {
            if (!detail?.filename) return;
            for (const n of findPreviewNodes()) setSource(n, detail);
        });

        api.addEventListener("executed", ({ detail }) => {
            const id = detail?.node;
            if (id == null) return;
            const node = app.graph.getNodeById(id);
            if (!node || node.type !== NODE_TYPE) return;
            const videos = detail?.output?.penplotter_videos;
            if (Array.isArray(videos) && videos.length > 0) {
                setSource(node, videos[0]);
            }
        });
    },
});
