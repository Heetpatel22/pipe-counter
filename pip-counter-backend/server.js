const cors = require("cors");
const express = require("express");
const multer = require("multer");
const ort = require("onnxruntime-node");
const path = require("path");
const sharp = require("sharp");
const { execFile } = require("child_process");
const fs = require("fs");

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

const PORT = process.env.PORT || 5005;
const DEFAULT_IMAGE_SIZE = 640;
const DEFAULT_IOU_THRESHOLD = 0.45;

const STREAMS = {
  hollow: {
    label: "Hollow Pipe",
    category: "Hollow Pipe Stack",
    modelPath: path.resolve(__dirname, "..", "hollow_specific", "best_hollow.onnx"),
    defaultConfidence: 0.25,
    ringColor: "#00c850",
  },
  solid: {
    label: "Solid Pipe",
    category: "Solid Steel Rod Stack",
    modelPath: path.resolve(__dirname, "..", "solid_specific", "best_solid.onnx"),
    defaultConfidence: 0.6,
    ringColor: "#1a368c",
  },
};

const sessions = {};
const UI_BUILD_DIR = path.resolve(__dirname, "..", "pipe-counter-ui", "build");
const SAMPLES_DIR = path.resolve(__dirname, "..", "Sample_images");

app.use(cors());
app.use(express.json());
app.use("/samples", express.static(SAMPLES_DIR));

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function intersectionOverUnion(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const aArea = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const bArea = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);

  return intersection / (aArea + bArea - intersection || 1);
}

function nonMaxSuppression(boxes, iouThreshold) {
  const sorted = [...boxes].sort((a, b) => b.confidence - a.confidence);
  const selected = [];

  while (sorted.length > 0) {
    const current = sorted.shift();
    selected.push(current);

    for (let index = sorted.length - 1; index >= 0; index -= 1) {
      if (intersectionOverUnion(current, sorted[index]) > iouThreshold) {
        sorted.splice(index, 1);
      }
    }
  }

  return selected;
}

function getPredictionValue(data, dims, detectionIndex, attributeIndex) {
  const numAttributes = dims[1];
  const numDetections = dims[2];

  if (numAttributes < numDetections) {
    return data[attributeIndex * numDetections + detectionIndex];
  }

  return data[detectionIndex * numAttributes + attributeIndex];
}

function decodeYoloOutput(outputTensor, confidenceThreshold, iouThreshold, imageSize, originalWidth, originalHeight) {
  const data = outputTensor.data;
  const dims = outputTensor.dims;

  if (dims.length !== 3) {
    throw new Error(`Unexpected YOLO output shape: ${dims.join("x")}`);
  }

  const numAttributes = Math.min(dims[1], dims[2]);
  const numDetections = Math.max(dims[1], dims[2]);
  const scaleX = originalWidth / imageSize;
  const scaleY = originalHeight / imageSize;
  const candidates = [];

  for (let i = 0; i < numDetections; i += 1) {
    let bestConfidence = 0;
    let classId = 0;

    for (let j = 4; j < numAttributes; j += 1) {
      const confidence = getPredictionValue(data, dims, i, j);
      if (confidence > bestConfidence) {
        bestConfidence = confidence;
        classId = j - 4;
      }
    }

    if (bestConfidence < confidenceThreshold) {
      continue;
    }

    const cx = getPredictionValue(data, dims, i, 0);
    const cy = getPredictionValue(data, dims, i, 1);
    const width = getPredictionValue(data, dims, i, 2);
    const height = getPredictionValue(data, dims, i, 3);

    candidates.push({
      x1: clamp((cx - width / 2) * scaleX, 0, originalWidth),
      y1: clamp((cy - height / 2) * scaleY, 0, originalHeight),
      x2: clamp((cx + width / 2) * scaleX, 0, originalWidth),
      y2: clamp((cy + height / 2) * scaleY, 0, originalHeight),
      confidence: bestConfidence,
      classId,
    });
  }

  return nonMaxSuppression(candidates, iouThreshold);
}

async function preprocessImage(fileBuffer, imageSize, applyGlareFilter) {
  let pipeline = sharp(fileBuffer).rotate();

  if (applyGlareFilter) {
    pipeline = pipeline.normalise();
  }

  const metadata = await pipeline.metadata();
  const originalWidth = metadata.width || imageSize;
  const originalHeight = metadata.height || imageSize;

  const { data } = await pipeline
    .resize(imageSize, imageSize, { fit: "fill" })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const planeSize = imageSize * imageSize;
  const input = new Float32Array(3 * planeSize);

  for (let i = 0; i < planeSize; i += 1) {
    input[i] = data[i * 3] / 255;
    input[i + planeSize] = data[i * 3 + 1] / 255;
    input[i + 2 * planeSize] = data[i * 3 + 2] / 255;
  }

  return {
    tensor: new ort.Tensor("float32", input, [1, 3, imageSize, imageSize]),
    originalWidth,
    originalHeight,
  };
}

async function getGrayscalePixels(fileBuffer, applyGlareFilter) {
  let pipeline = sharp(fileBuffer).rotate();

  if (applyGlareFilter) {
    pipeline = pipeline.normalise();
  }

  const { data, info } = await pipeline.grayscale().raw().toBuffer({ resolveWithObject: true });
  return { data, width: info.width, height: info.height };
}

function countHollowCenters(gray, boxes) {
  let hollowCount = 0;

  for (const box of boxes) {
    const xmin = Math.floor(clamp(box.x1, 0, gray.width - 1));
    const ymin = Math.floor(clamp(box.y1, 0, gray.height - 1));
    const xmax = Math.ceil(clamp(box.x2, 0, gray.width));
    const ymax = Math.ceil(clamp(box.y2, 0, gray.height));
    const width = xmax - xmin;
    const height = ymax - ymin;

    if (width <= 1 || height <= 1) {
      continue;
    }

    const centerX = xmin + width / 2;
    const centerY = ymin + height / 2;
    const radius = Math.min(width, height) * 0.25;
    let centerSum = 0;
    let centerPixels = 0;
    let outerSum = 0;
    let outerPixels = 0;

    for (let y = ymin; y < ymax; y += 1) {
      for (let x = xmin; x < xmax; x += 1) {
        const value = gray.data[y * gray.width + x];
        const distance = Math.hypot(x - centerX, y - centerY);

        if (distance <= radius) {
          centerSum += value;
          centerPixels += 1;
        } else {
          outerSum += value;
          outerPixels += 1;
        }
      }
    }

    if (centerPixels === 0 || outerPixels === 0) {
      continue;
    }

    const centerBrightness = centerSum / centerPixels;
    const outerBrightness = outerSum / outerPixels;

    if (centerBrightness < outerBrightness * 0.65 && centerBrightness < 100) {
      hollowCount += 1;
    }
  }

  return hollowCount;
}

function evaluateBatch(pipeType, stream, boxes, avgConfidence, hollowCount) {
  const total = boxes.length;
  const hollowRatio = total > 0 ? hollowCount / total : 0;

  if (total === 0) {
    return {
      accepted: false,
      invalidDetection: true,
      status: "Invalid detection",
      category: stream.category,
      count: 0,
      avgConfidence: 0,
      reason: `No ${stream.label.toLowerCase()} targets were detected above the confidence threshold.`,
    };
  }

  if (pipeType === "solid" && hollowRatio >= 0.10) {
    return {
      accepted: false,
      invalidDetection: true,
      status: "Invalid detection",
      category: "Hollow Pipe Detected",
      count: 0,
      avgConfidence,
      reason: `Selected solid pipe, but hollow pipe geometry was detected (${(hollowRatio * 100).toFixed(1)}% targets flagged as hollow). Limit 10.0%.`,
    };
  }

  if (pipeType === "hollow" && hollowRatio < 0.2) {
    return {
      accepted: false,
      invalidDetection: true,
      status: "Invalid detection",
      category: "Solid Pipe Detected",
      count: 0,
      avgConfidence,
      reason: "Selected hollow pipe, but this batch does not meet the hollow-pipe verification rule.",
    };
  }

  return {
    accepted: true,
    invalidDetection: false,
    status: "Accepted",
    category: stream.category,
    count: total,
    avgConfidence,
    reason: `${stream.label} batch accepted.`,
  };
}

async function createAnnotatedImage(fileBuffer, boxes, color) {
  const metadata = await sharp(fileBuffer).rotate().metadata();
  const width = metadata.width || DEFAULT_IMAGE_SIZE;
  const height = metadata.height || DEFAULT_IMAGE_SIZE;

  const circles = boxes
    .map((box) => {
      const cx = (box.x1 + box.x2) / 2;
      const cy = (box.y1 + box.y2) / 2;
      const radius = Math.max(box.x2 - box.x1, box.y2 - box.y1) / 2;

      return `<circle cx="${cx}" cy="${cy}" r="${radius}" fill="none" stroke="${color}" stroke-width="4" />`;
    })
    .join("");

  const overlay = Buffer.from(
    `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">${circles}</svg>`
  );

  return sharp(fileBuffer)
    .rotate()
    .composite([{ input: overlay, top: 0, left: 0 }])
    .jpeg({ quality: 88 })
    .toBuffer();
}

app.get("/health", (req, res) => {
  res.json({
    ok: true,
    streams: Object.fromEntries(
      Object.entries(STREAMS).map(([key, stream]) => [key, { modelLoaded: Boolean(sessions[key]), modelPath: stream.modelPath }])
    ),
  });
});

app.post("/detect", upload.single("image"), async (req, res) => {
  const pipeType = req.body.pipeType === "solid" ? "solid" : "hollow";
  const stream = STREAMS[pipeType];
  const session = sessions[pipeType];

  if (!session) {
    return res.status(503).json({ success: false, error: `${stream.label} model is still loading.` });
  }

  if (!req.file) {
    return res.status(400).json({ success: false, error: "No image uploaded." });
  }

  const tempFilePath = path.join(__dirname, "..", `temp_${Date.now()}.jpg`);

  try {
    const confidenceThreshold = clamp(
      Number(req.body.confidence) || stream.defaultConfidence,
      0.05,
      1
    );
    const iouThreshold = clamp(Number(req.body.iou) || DEFAULT_IOU_THRESHOLD, 0.05, 1);
    const imageSize = clamp(Number(req.body.imageSize) || DEFAULT_IMAGE_SIZE, 320, 1280);
    const applyGlareFilter = req.body.glareFilter === "true";

    // Write temp file for the python script
    await fs.promises.writeFile(tempFilePath, req.file.buffer);

    // Run Gemini process and YOLO preprocessing in parallel
    const runGemini = new Promise((resolve) => {
      const pythonCmd = process.platform === "win32" ? "python" : "python3";
      const scriptPath = path.resolve(__dirname, "..", "gemini_process.py");
      execFile(pythonCmd, [scriptPath, tempFilePath], (error, stdout, stderr) => {
        if (error) {
          console.error(`Gemini process error: ${stderr || error.message}`);
          resolve("Failed to run Gemini glare-restoration analysis.");
        } else {
          resolve(stdout.trim());
        }
      });
    });

    const runYolo = Promise.all([
      preprocessImage(req.file.buffer, imageSize, applyGlareFilter),
      getGrayscalePixels(req.file.buffer, applyGlareFilter),
    ]);

    const [geminiText, [{ tensor, originalWidth, originalHeight }, gray]] = await Promise.all([
      runGemini,
      runYolo,
    ]);

    const outputs = await session.run({ [session.inputNames[0]]: tensor });
    const outputTensor = outputs[session.outputNames[0]];
    const boxes = decodeYoloOutput(
      outputTensor,
      confidenceThreshold,
      iouThreshold,
      imageSize,
      originalWidth,
      originalHeight
    );
    const avgConfidence =
      boxes.length > 0
        ? (boxes.reduce((sum, box) => sum + box.confidence, 0) / boxes.length) * 100
        : 0;
    const hollowCount = countHollowCenters(gray, boxes);
    const batch = evaluateBatch(pipeType, stream, boxes, avgConfidence, hollowCount);
    const annotatedImage = batch.accepted
      ? `data:image/jpeg;base64,${(await createAnnotatedImage(req.file.buffer, boxes, stream.ringColor)).toString("base64")}`
      : "";

    return res.json({
      success: true,
      pipeType,
      imageMode: applyGlareFilter ? "Normalized" : "Raw image",
      geminiAnalysis: geminiText,
      ...batch,
      annotatedImage,
    });
  } catch (error) {
    console.error(error);
    return res.status(500).json({ success: false, error: "Inference processing failed." });
  } finally {
    // Ensure temp file is cleaned up
    fs.promises.unlink(tempFilePath).catch((err) => {
      if (err.code !== "ENOENT") {
        console.error("Temp file cleanup failed:", err);
      }
    });
  }
});

app.use(express.static(UI_BUILD_DIR));

app.get(/.*/, (req, res) => {
  res.sendFile(path.join(UI_BUILD_DIR, "index.html"));
});

Promise.all(
  Object.entries(STREAMS).map(async ([key, stream]) => {
    sessions[key] = await ort.InferenceSession.create(stream.modelPath);
    console.log(`${stream.label} ONNX model loaded: ${stream.modelPath}`);
  })
)
  .then(() => {
    app.listen(PORT, "0.0.0.0", () => {
      console.log(`Backend running on http://localhost:${PORT}`);
    });
  })
  .catch((error) => {
    console.error("Failed to load ONNX models:", error);
    process.exit(1);
  });
