import {
    HandLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs";

const video = document.getElementById("webcam");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const sourceCanvas = document.createElement("canvas");
const sourceCtx = sourceCanvas.getContext("2d");

// Separate canvas used to capture the already-rendered image
// before applying the blur polygon.
const blurCanvas = document.createElement("canvas");
const blurCtx = blurCanvas.getContext("2d");

// Smaller canvas for expensive edge detection
sourceCanvas.width = 320;
sourceCanvas.height = 240;
let lastDetectionTime = 0;
const detectionInterval = 1000 / 30;

// ------------------------------------
// CAMERA
// ------------------------------------

const stream = await navigator.mediaDevices.getUserMedia({
    video: true
});

video.srcObject = stream;

await new Promise(resolve => {
    video.onloadeddata = resolve;
});

canvas.width = video.videoWidth;
canvas.height = video.videoHeight;

blurCanvas.width = canvas.width;
blurCanvas.height = canvas.height;
// ------------------------------------
// MEDIAPIPE
// ------------------------------------

const vision = await FilesetResolver.forVisionTasks(
  "./wasm"
);

const handLandmarker = await HandLandmarker.createFromOptions(vision, {
  baseOptions: {
    // CHANGE THIS LINE to point to your local file instead of the googleapis URL:
    modelAssetPath: "./hand_landmarker.task", 
    delegate: "GPU"
  },
  runningMode: "VIDEO",
  numHands: 2
});

// ------------------------------------
// STATE
// ------------------------------------

let polygonExists = false;
let previousTouch = false;

let blurPolygonExists = false;
let blurPreviousTouch = false;

let currentFilter = 0;
let previousIndexExtended = false;
// ------------------------------------
// DISTANCE
// ------------------------------------

function distance(a, b) {
    const dx = a.x - b.x;
    const dy = a.y - b.y;

    return Math.sqrt(dx * dx + dy * dy);
}

// ------------------------------------
// NORMAL FILTERS
// ------------------------------------

function applyCurrentFilter() {

    if (currentFilter === 0) {

        ctx.filter = "grayscale(100%) contrast(500%)";

    } else if (currentFilter === 1) {

        ctx.filter = "invert(100%)";

    } else if (currentFilter === 2) {

        ctx.filter = "sepia(100%)";

    }
}

// ------------------------------------
// GREEN EDGE FILTER
// ------------------------------------

function drawGreenEdges() {

    sourceCtx.drawImage(
        video,
        0,
        0,
        sourceCanvas.width,
        sourceCanvas.height
    );

    const imageData = sourceCtx.getImageData(
        0,
        0,
        sourceCanvas.width,
        sourceCanvas.height
    );

    const pixels = imageData.data;

    const output = sourceCtx.createImageData(
        sourceCanvas.width,
        sourceCanvas.height
    );

    const outputPixels = output.data;

    const width = sourceCanvas.width;
    const height = sourceCanvas.height;

    for (let y = 1; y < height - 1; y++) {

        for (let x = 1; x < width - 1; x++) {

            const i = (y * width + x) * 4;

            const right = i + 4;
            const below = i + width * 4;

            const currentBrightness =
                pixels[i] +
                pixels[i + 1] +
                pixels[i + 2];

            const rightBrightness =
                pixels[right] +
                pixels[right + 1] +
                pixels[right + 2];

            const belowBrightness =
                pixels[below] +
                pixels[below + 1] +
                pixels[below + 2];

            const difference =
                Math.abs(
                    currentBrightness -
                    rightBrightness
                ) +
                Math.abs(
                    currentBrightness -
                    belowBrightness
                );

            if (difference > 40) {

                outputPixels[i] = 0;
                outputPixels[i + 1] = 255;
                outputPixels[i + 2] = 0;
                outputPixels[i + 3] = 255;

            } else {

                outputPixels[i] = 0;
                outputPixels[i + 1] = 0;
                outputPixels[i + 2] = 0;
                outputPixels[i + 3] = 255;
            }
        }
    }

    sourceCtx.putImageData(
        output,
        0,
        0
    );

    ctx.drawImage(
        sourceCanvas,
        0,
        0,
        canvas.width,
        canvas.height
    );
}
function drawHardBlackWhite() {

    // Put webcam into the small processing canvas
    sourceCtx.drawImage(
        video,
        0,
        0,
        sourceCanvas.width,
        sourceCanvas.height
    );

    const imageData = sourceCtx.getImageData(
        0,
        0,
        sourceCanvas.width,
        sourceCanvas.height
    );

    const pixels = imageData.data;

    const output = sourceCtx.createImageData(
        sourceCanvas.width,
        sourceCanvas.height
    );

    const outputPixels = output.data;

    const width = sourceCanvas.width;
    const height = sourceCanvas.height;

    for (let y = 0; y < height; y++) {

        for (let x = 0; x < width; x++) {

            const i = (y * width + x) * 4;

            // Convert pixel to brightness
            const brightness =
                0.299 * pixels[i] +
                0.587 * pixels[i + 1] +
                0.114 * pixels[i + 2];

            // Adjust this number to control the threshold
            const threshold = 100;

            if (brightness > threshold) {

                // WHITE
                outputPixels[i] = 255;
                outputPixels[i + 1] = 255;
                outputPixels[i + 2] = 255;

            } else {

                // BLACK
                outputPixels[i] = 0;
                outputPixels[i + 1] = 0;
                outputPixels[i + 2] = 0;
            }

            outputPixels[i + 3] = 255;
        }
    }

    sourceCtx.putImageData(
        output,
        0,
        0
    );

    // Draw processed image into the current polygon clip
    ctx.drawImage(
        sourceCanvas,
        0,
        0,
        canvas.width,
        canvas.height
    );
}
function drawRainbow() {

    // Put webcam into the small processing canvas
    sourceCtx.drawImage(
        video,
        0,
        0,
        sourceCanvas.width,
        sourceCanvas.height
    );

    const imageData = sourceCtx.getImageData(
        0,
        0,
        sourceCanvas.width,
        sourceCanvas.height
    );

    const pixels = imageData.data;

    const output = sourceCtx.createImageData(
        sourceCanvas.width,
        sourceCanvas.height
    );

    const outputPixels = output.data;

    const width = sourceCanvas.width;
    const height = sourceCanvas.height;

    for (let y = 0; y < height; y++) {

        for (let x = 0; x < width; x++) {

            const i = (y * width + x) * 4;

            // Original brightness
            const brightness =
                0.299 * pixels[i] +
                0.587 * pixels[i + 1] +
                0.114 * pixels[i + 2];

            // Turn brightness into a rainbow hue
            const hue =
                (brightness * 2 + x * 2 + y * 1.5) % 360;
            const saturation = 100;
            const lightness = brightness / 255 * 50;

            // HSV/HSL conversion
            const c =
                (1 - Math.abs(2 * lightness / 100 - 1))
                * saturation / 100;

            const h =
                hue / 60;

            const xColor =
                c * (1 - Math.abs(h % 2 - 1));

            let r = 0;
            let g = 0;
            let b = 0;

            if (h < 1) {
                r = c;
                g = xColor;
            } else if (h < 2) {
                r = xColor;
                g = c;
            } else if (h < 3) {
                g = c;
                b = xColor;
            } else if (h < 4) {
                g = xColor;
                b = c;
            } else if (h < 5) {
                r = xColor;
                b = c;
            } else {
                r = c;
                b = xColor;
            }

            const m =
                lightness / 100 - c / 2;

            outputPixels[i] =
                (r + m) * 255;

            outputPixels[i + 1] =
                (g + m) * 255;

            outputPixels[i + 2] =
                (b + m) * 255;

            outputPixels[i + 3] = 255;
        }
    }

    sourceCtx.putImageData(
        output,
        0,
        0
    );

    ctx.drawImage(
        sourceCanvas,
        0,
        0,
        canvas.width,
        canvas.height
    );
}




function getIntersection(a, b, c, d) {

    const denominator =
        (a.x - b.x) * (c.y - d.y) -
        (a.y - b.y) * (c.x - d.x);

    if (denominator === 0) {
        return null;
    }

    const t =
        (
            (a.x - c.x) * (c.y - d.y) -
            (a.y - c.y) * (c.x - d.x)
        ) / denominator;

    const u =
        -(
            (a.x - b.x) * (a.y - c.y) -
            (a.y - b.y) * (a.x - c.x)
        ) / denominator;

    if (
        t < 0 ||
        t > 1 ||
        u < 0 ||
        u > 1
    ) {
        return null;
    }

    return {
        x: a.x + t * (b.x - a.x),
        y: a.y + t * (b.y - a.y)
    };
}



function drawSelectedFilter() {

    ctx.filter = "none";

    if (currentFilter === 0) {

        drawHardBlackWhite();

    } else if (currentFilter === 1) {

        ctx.filter = "invert(100%)";

        ctx.drawImage(
            video,
            0,
            0,
            canvas.width,
            canvas.height
        );

        ctx.filter = "none";

    } else if (currentFilter === 2) {

        drawRainbow();

    } else if (currentFilter === 3) {

        drawGreenEdges();
    }

    ctx.filter = "none";
}

function detectHands(timestamp) {

    if (
        timestamp - lastDetectionTime <
        detectionInterval
    ) {
        requestAnimationFrame(detectHands);
        return;
    }

    lastDetectionTime = timestamp;

    const results =
        handLandmarker.detectForVideo(
            video,
            timestamp
        );

    const hands =
        results.landmarks ?? [];


    ctx.filter = "none";

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    ctx.drawImage(
        video,
        0,
        0,
        canvas.width,
        canvas.height
    );


    if (hands.length === 2) {

        const leftIndex =
            results.handedness.findIndex(
                hand =>
                    hand[0].categoryName === "Left"
            );

        const rightIndex =
            results.handedness.findIndex(
                hand =>
                    hand[0].categoryName === "Right"
            );

        if (
            leftIndex === -1 ||
            rightIndex === -1
        ) {
            requestAnimationFrame(detectHands);
            return;
        }

        const hand1 = hands[leftIndex];
        const hand2 = hands[rightIndex];

        const rightIndexExtended =
            hand2[8].y < hand2[6].y;

        if (
            polygonExists &&
            rightIndexExtended &&
            !previousIndexExtended
        ) {

            currentFilter++;

            if (currentFilter > 3) {
                currentFilter = 0;
            }
        }


        previousIndexExtended =
            rightIndexExtended;


        const thumb1 = hand1[4];
        const middle1 = hand1[12];
        const index1 = hand1[8];

        const thumb2 = hand2[4];
        const middle2 = hand2[12];
        const index2 = hand2[8];

        // ------------------------------------
        // CREATE / DELETE POLYGON
        // ------------------------------------

        // ------------------------------------
        // CREATE / DELETE POLYGON
        // ------------------------------------

        // ------------------------------------
        // CREATE / DELETE POLYGONS
        // ------------------------------------

        // THUMB + MIDDLE = NORMAL POLYGON

        const leftMiddlePinching =
            distance(
                thumb1,
                middle1
            ) < 0.08;

        const rightMiddlePinching =
            distance(
                thumb2,
                middle2
            ) < 0.08;

        const normalPinching =
            leftMiddlePinching &&
            rightMiddlePinching;

        if (
            normalPinching &&
            !previousTouch
        ) {

            polygonExists =
                !polygonExists;
        }


        // THUMB + INDEX = BLUR POLYGON

        const leftIndexPinching =
            distance(
                thumb1,
                index1
            ) < 0.08;

        const rightIndexPinching =
            distance(
                thumb2,
                index2
            ) < 0.08;

        const blurPinching =
            leftIndexPinching &&
            rightIndexPinching;

        if (
            blurPinching &&
            !blurPreviousTouch
        ) {

            blurPolygonExists =
                !blurPolygonExists;
        }


        // Remember previous pinch states

        previousTouch =
            normalPinching;

        blurPreviousTouch =
            blurPinching;

        const p1 = {
            x: thumb1.x * canvas.width,
            y: thumb1.y * canvas.height
        };

        const p2 = {
            x: middle1.x * canvas.width,
            y: middle1.y * canvas.height
        };

        const p3 = {
            x: middle2.x * canvas.width,
            y: middle2.y * canvas.height
        };

        const p4 = {
            x: thumb2.x * canvas.width,
            y: thumb2.y * canvas.height
        };


        // ------------------------------------
        // BLUR POLYGON POINTS
        // ------------------------------------

        const blurP1 = {
            x: thumb1.x * canvas.width,
            y: thumb1.y * canvas.height
        };

        const blurP2 = {
            x: index1.x * canvas.width,
            y: index1.y * canvas.height
        };

        const blurP3 = {
            x: index2.x * canvas.width,
            y: index2.y * canvas.height
        };

        const blurP4 = {
            x: thumb2.x * canvas.width,
            y: thumb2.y * canvas.height
        };


        // ------------------------------------
        // POLYGON
        // ------------------------------------
        // ------------------------------------
        // POLYGON
        // ------------------------------------

        // ------------------------------------
        // NORMAL POLYGON
        // ------------------------------------

        if (polygonExists) {

            const intersection1 =
                getIntersection(
                    p1,
                    p2,
                    p3,
                    p4
                );

            const intersection2 =
                getIntersection(
                    p2,
                    p3,
                    p4,
                    p1
                );

            // ====================================
            // NORMAL POLYGON
            // ====================================

            if (
                !intersection1 &&
                !intersection2
            ) {

                ctx.save();

                ctx.beginPath();

                ctx.moveTo(
                    p1.x,
                    p1.y
                );

                ctx.lineTo(
                    p2.x,
                    p2.y
                );

                ctx.lineTo(
                    p3.x,
                    p3.y
                );

                ctx.lineTo(
                    p4.x,
                    p4.y
                );

                ctx.closePath();

                ctx.clip();

                drawSelectedFilter();

                ctx.restore();
            }

            // ====================================
            // HOURGLASS
            // ====================================

            else {

                const cross =
                    intersection1 ||
                    intersection2;

                if (intersection1) {

                    // FIRST TRIANGLE

                    ctx.save();

                    ctx.beginPath();

                    ctx.moveTo(
                        p1.x,
                        p1.y
                    );

                    ctx.lineTo(
                        cross.x,
                        cross.y
                    );

                    ctx.lineTo(
                        p4.x,
                        p4.y
                    );

                    ctx.closePath();

                    ctx.clip();

                    drawSelectedFilter();

                    ctx.restore();


                    // SECOND TRIANGLE
                    // Always green edges

                    ctx.save();

                    ctx.beginPath();

                    ctx.moveTo(
                        p2.x,
                        p2.y
                    );

                    ctx.lineTo(
                        cross.x,
                        cross.y
                    );

                    ctx.lineTo(
                        p3.x,
                        p3.y
                    );

                    ctx.closePath();

                    ctx.clip();

                    drawGreenEdges();

                    ctx.restore();

                } else {

                    // FIRST TRIANGLE

                    ctx.save();

                    ctx.beginPath();

                    ctx.moveTo(
                        p1.x,
                        p1.y
                    );

                    ctx.lineTo(
                        p2.x,
                        p2.y
                    );

                    ctx.lineTo(
                        cross.x,
                        cross.y
                    );

                    ctx.closePath();

                    ctx.clip();

                    drawSelectedFilter();

                    ctx.restore();


                    // SECOND TRIANGLE
                    // Always green edges

                    ctx.save();

                    ctx.beginPath();

                    ctx.moveTo(
                        cross.x,
                        cross.y
                    );

                    ctx.lineTo(
                        p3.x,
                        p3.y
                    );

                    ctx.lineTo(
                        p4.x,
                        p4.y
                    );

                    ctx.closePath();

                    ctx.clip();

                    drawGreenEdges();

                    ctx.restore();
                }
            }

            // ------------------------------------
            // NORMAL POLYGON OUTLINE
            // ------------------------------------

            ctx.filter = "none";

            ctx.beginPath();

            ctx.moveTo(
                p1.x,
                p1.y
            );

            ctx.lineTo(
                p2.x,
                p2.y
            );

            ctx.lineTo(
                p3.x,
                p3.y
            );

            ctx.lineTo(
                p4.x,
                p4.y
            );

            ctx.closePath();

            ctx.stroke();
        }


        // ====================================
        // BLUR POLYGON
        // ====================================

        if (blurPolygonExists) {

            // Copy the CURRENTLY rendered canvas.
            // This makes blur affect everything underneath it.

            blurCtx.clearRect(
                0,
                0,
                blurCanvas.width,
                blurCanvas.height
            );

            blurCtx.drawImage(
                canvas,
                0,
                0
            );

            ctx.save();

            ctx.beginPath();

            ctx.moveTo(
                blurP1.x,
                blurP1.y
            );

            ctx.lineTo(
                blurP2.x,
                blurP2.y
            );

            ctx.lineTo(
                blurP3.x,
                blurP3.y
            );

            ctx.lineTo(
                blurP4.x,
                blurP4.y
            );

            ctx.closePath();

            ctx.clip();

          // --------------------------------
// PIXELATION
// --------------------------------

// Small temporary canvas.
// Smaller = bigger pixels.
const pixelSize = 12;

const pixelCanvas = document.createElement("canvas");
const pixelCtx = pixelCanvas.getContext("2d");

pixelCanvas.width =
    Math.max(1, Math.floor(canvas.width / pixelSize));

pixelCanvas.height =
    Math.max(1, Math.floor(canvas.height / pixelSize));

// Turn off smoothing so the enlarged image has hard pixel edges.
pixelCtx.imageSmoothingEnabled = false;
ctx.imageSmoothingEnabled = false;

// Shrink the rendered image.
pixelCtx.drawImage(
    blurCanvas,
    0,
    0,
    pixelCanvas.width,
    pixelCanvas.height
);

// Stretch it back up.
ctx.drawImage(
    pixelCanvas,
    0,
    0,
    pixelCanvas.width,
    pixelCanvas.height,
    0,
    0,
    canvas.width,
    canvas.height
);

            ctx.restore();
        }
    }

    // ------------------------------------
    // HAND SKELETON
    // ------------------------------------

    ctx.filter = "none";

    for (const hand of hands) {

        const connections = [
            [0, 1], [1, 2], [2, 3], [3, 4],
            [0, 5], [5, 6], [6, 7], [7, 8],
            [0, 9], [9, 10], [10, 11], [11, 12],
            [0, 13], [13, 14], [14, 15], [15, 16],
            [0, 17], [17, 18], [18, 19], [19, 20],
            [5, 9], [9, 13], [13, 17]
        ];

        // Connections
        for (const [a, b] of connections) {

            const x1 =
                hand[a].x *
                canvas.width;

            const y1 =
                hand[a].y *
                canvas.height;

            const x2 =
                hand[b].x *
                canvas.width;

            const y2 =
                hand[b].y *
                canvas.height;

            ctx.beginPath();

            ctx.moveTo(
                x1,
                y1
            );

            ctx.lineTo(
                x2,
                y2
            );

            ctx.stroke();
        }

        // Landmarks
        for (const point of hand) {

            const x =
                point.x *
                canvas.width;

            const y =
                point.y *
                canvas.height;

            ctx.beginPath();

            ctx.arc(
                x,
                y,
                7,
                0,
                Math.PI * 2
            );

            ctx.fill();
        }
    }

    requestAnimationFrame(detectHands);
}

requestAnimationFrame(detectHands);
