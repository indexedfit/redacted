// Offline-first app using local, self-hosted MediaPipe Tasks + WASM + model
const TASKS_VERSION = "0.10.14";

// Import the local ESM bundle we ship (converted to .js for MIME friendliness on common servers)
const { FaceDetector, FilesetResolver } = await import("./lib/tasks-vision.js");

// WASM assets - detect SIMD support and load appropriate variant
async function getWasmBase() {
  // Test SIMD support with a minimal WASM module
  const simdTest = new Uint8Array([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x05, 0x01, 0x60,
    0x00, 0x01, 0x7b, 0x03, 0x02, 0x01, 0x00, 0x0a, 0x0a, 0x01, 0x08, 0x00,
    0xfd, 0x0f, 0xfd, 0x62, 0x0b
  ]);
  
  try {
    await WebAssembly.instantiate(simdTest);
    console.log('SIMD supported - using optimized WASM');
    return { 
      wasmLoaderScript: './wasm/vision_wasm_internal.js',
      wasmBinary: './wasm/vision_wasm_internal.wasm'
    };
  } catch {
    console.log('SIMD not supported - using fallback WASM');
    return { 
      wasmLoaderScript: './wasm/vision_wasm_nosimd_internal.js',
      wasmBinary: './wasm/vision_wasm_nosimd_internal.wasm'
    };
  }
}

const WASM_CONFIG = await getWasmBase();

// Model is served locally from ./models on first run; then cached in OPFS
const MODEL_PATH = "./models/face_detector.tflite";

// UI
const fileInput = document.getElementById('file');
const detectBtn = document.getElementById('detectBtn');
const resetBtn = document.getElementById('resetBtn');
const downloadBtn = document.getElementById('downloadBtn');
const statusEl = document.getElementById('status');
const canvas = document.getElementById('stage');
const ctx = canvas.getContext('2d', { willReadFrequently: true });

let detector = null;
let img = new Image();
let imgURL = null;
let detections = [];
let originalImageData = null;
let redactionBoxes = [];
let draggedBox = null;
let dragOffset = { x: 0, y: 0 };

function setStatus(msg, isError=false) {
  statusEl.textContent = msg;
  statusEl.className = isError ? 'status error' : 'status';
  console.log('[status]', msg);
}

// OPFS loader (first try OPFS, otherwise fetch local file and persist)
async function loadModelBufferOPFS({ path = MODEL_PATH, filename = "face_detector.tflite" } = {}) {
  try {
    const root = await navigator.storage.getDirectory();
    try {
      const fh = await root.getFileHandle(filename);
      const file = await fh.getFile();
      return new Uint8Array(await file.arrayBuffer());
    } catch {
      const resp = await fetch(path, { cache: 'force-cache' });
      if (!resp.ok) throw new Error('Model fetch failed: ' + resp.status);
      const buf = await resp.arrayBuffer();
      const newHandle = await root.getFileHandle(filename, { create: true });
      const w = await newHandle.createWritable();
      await w.write(buf);
      await w.close();
      return new Uint8Array(buf);
    }
  } catch (e) {
    console.warn('OPFS unavailable or failed, falling back to local fetch only.', e);
    const resp = await fetch(path, { cache: 'force-cache' });
    if (!resp.ok) throw new Error('Model fetch failed: ' + resp.status);
    return new Uint8Array(await resp.arrayBuffer());
  }
}

async function initDetector() {
  setStatus('Loading model...');
  // Use the detected WASM configuration
  const fileset = await FilesetResolver.forVisionTasks("./wasm");
  const modelBuffer = await loadModelBufferOPFS();
  detector = await FaceDetector.createFromOptions(fileset, {
    runningMode: "IMAGE",
    baseOptions: { modelAssetBuffer: modelBuffer }
  });
  setStatus('Ready');
}

await initDetector();

function resetState() {
  detections = [];
  redactionBoxes = [];
  if (originalImageData) {
    ctx.putImageData(originalImageData, 0, 0);
  } else {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
  downloadBtn.disabled = true;
  setStatus('Ready');
}

resetBtn.addEventListener('click', resetState);

// Load image
fileInput.addEventListener('change', (e) => {
  const file = e.target.files && e.target.files[0];
  if (!file) return;
  if (imgURL) URL.revokeObjectURL(imgURL);
  imgURL = URL.createObjectURL(file);

  img.onload = () => {
    const maxDim = 4096;
    const scale = Math.min(1, maxDim / Math.max(img.naturalWidth, img.naturalHeight));
    const w = Math.round(img.naturalWidth * scale);
    const h = Math.round(img.naturalHeight * scale);
    canvas.width = w;
    canvas.height = h;
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(img, 0, 0, w, h);
    originalImageData = ctx.getImageData(0, 0, w, h);
    detections = [];
    redactionBoxes = [];

    detectBtn.disabled = false;
    resetBtn.disabled = false;
    downloadBtn.disabled = true;
    setStatus('Image loaded');
  };
  img.onerror = () => setStatus('Error loading image', true);
  img.src = imgURL;
});

// Detect and Redact
detectBtn.addEventListener('click', async () => {
  if (!detector) return setStatus('Detector not initialized.', true);
  if (canvas.width === 0 || canvas.height === 0) return;
  try {
    setStatus('Detecting faces...');
    const result = await detector.detect(canvas);
    detections = result?.detections ?? [];
    console.log('Detections:', detections);
    
    if (detections.length === 0) {
      setStatus('No faces detected');
      return;
    }
    
    // Restore original image
    ctx.putImageData(originalImageData, 0, 0);
    
    // Calculate and draw redaction boxes
    calculateRedactionBoxes();
    
    if (redactionBoxes.length === 0) {
      console.error('No redaction boxes were created!');
      setStatus('Error creating redaction boxes', true);
      return;
    }
    
    drawRedactionBoxes();
    
    downloadBtn.disabled = false;
    setStatus(`Redacted ${detections.length} face${detections.length === 1 ? '' : 's'}`);
  } catch (e) {
    console.error(e);
    setStatus('Error processing image', true);
  }
});

function drawDetections(dets) {
  ctx.save();
  ctx.lineWidth = Math.max(2, Math.round(canvas.width / 400));
  ctx.strokeStyle = '#10a37f';
  ctx.fillStyle = '#10a37f';

  for (const det of dets) {
    const bb = det.boundingBox;
    if (bb) ctx.strokeRect(bb.originX, bb.originY, bb.width, bb.height);
    if (det.keypoints) {
      for (const kp of det.keypoints) {
        ctx.beginPath();
        ctx.arc(kp.x, kp.y, Math.max(2, Math.round(canvas.width / 600)), 0, 2*Math.PI);
        ctx.fill();
      }
    }
  }
  ctx.restore();
}

// Calculate redaction boxes from detections
function calculateRedactionBoxes() {
  redactionBoxes = [];
  
  for (const det of detections) {
    // Try to use keypoints if available (indices 0 and 1 are the eyes)
    if (det.keypoints && det.keypoints.length >= 2) {
      const rightEye = det.keypoints[0]; // Right eye (from viewer's perspective)
      const leftEye = det.keypoints[1];  // Left eye
      
      if (rightEye && leftEye && rightEye.x !== undefined && leftEye.x !== undefined) {
        
        // Check if coordinates are normalized (0-1) or already in pixels
        const rightEyeX = rightEye.x < 2 ? rightEye.x * canvas.width : rightEye.x;
        const rightEyeY = rightEye.y < 2 ? rightEye.y * canvas.height : rightEye.y;
        const leftEyeX = leftEye.x < 2 ? leftEye.x * canvas.width : leftEye.x;
        const leftEyeY = leftEye.y < 2 ? leftEye.y * canvas.height : leftEye.y;
        
        // Calculate angle between eyes
        const dx = leftEyeX - rightEyeX;
        const dy = leftEyeY - rightEyeY;
        const angle = Math.atan2(dy, dx);
        const distance = Math.hypot(dx, dy);
        
        // Calculate center point between eyes
        const centerX = (rightEyeX + leftEyeX) / 2;
        const centerY = (rightEyeY + leftEyeY) / 2;
        
        // Make the bar wider to ensure full coverage
        const barWidth = distance * 2.5; // 250% of eye distance for full coverage
        const barHeight = distance * 0.5; // 50% of eye distance - thinner bars
        
        const box = {
          x: centerX,
          y: centerY,
          width: barWidth,
          height: barHeight,
          angle: angle, // Store angle for rotated drawing
          type: 'rotated'
        };
        
        redactionBoxes.push(box);
        continue;
      }
    }
    
    // Fallback to bounding box approach
    if (det.boundingBox) {
      const bb = det.boundingBox;
      const eyeY = bb.originY + bb.height * 0.3;
      const eyeWidth = bb.width * 0.8;
      const eyeHeight = bb.height * 0.2;
      
      const box = {
        x: bb.originX + bb.width * 0.1,
        y: eyeY - eyeHeight / 2,
        width: eyeWidth,
        height: eyeHeight,
        type: 'standard'
      };
      
      redactionBoxes.push(box);
    }
  }
  
  console.log('Total redaction boxes created:', redactionBoxes.length);
}

// Draw redaction boxes
function drawRedactionBoxes() {
  ctx.save();
  
  for (const box of redactionBoxes) {
    
    if (box.type === 'rotated') {
      // Draw rotated rectangle
      ctx.save();
      ctx.translate(box.x, box.y);
      ctx.rotate(box.angle);
      
      // Fill black rectangle
      ctx.fillStyle = '#000';
      ctx.fillRect(-box.width / 2, -box.height / 2, box.width, box.height);
      
      // Debug: draw border to see box bounds
      if (window.debugMode) {
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.strokeRect(-box.width / 2, -box.height / 2, box.width, box.height);
      }
      
      ctx.restore();
    } else {
      // Draw standard rectangle
      ctx.fillStyle = '#000';
      ctx.fillRect(box.x, box.y, box.width, box.height);
    }
  }
  
  ctx.restore();
}


// Download functionality
downloadBtn.addEventListener('click', () => {
  canvas.toBlob((blob) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'redacted-image.png';
    a.click();
    URL.revokeObjectURL(url);
    setStatus('Downloaded');
  });
});

// Mouse/touch interaction helpers
function getPointerPos(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  
  const clientX = e.touches ? e.touches[0].clientX : e.clientX;
  const clientY = e.touches ? e.touches[0].clientY : e.clientY;
  
  return {
    x: (clientX - rect.left) * scaleX,
    y: (clientY - rect.top) * scaleY
  };
}

function findBoxAt(x, y) {
  // Check boxes in reverse order (top to bottom)
  for (let i = redactionBoxes.length - 1; i >= 0; i--) {
    const box = redactionBoxes[i];
    
    if (box.type === 'rotated') {
      // For rotated boxes, check if point is within rotated rectangle
      // Transform point to box's local coordinates
      const dx = x - box.x;
      const dy = y - box.y;
      const cos = Math.cos(-box.angle);
      const sin = Math.sin(-box.angle);
      const localX = dx * cos - dy * sin;
      const localY = dx * sin + dy * cos;
      
      if (Math.abs(localX) <= box.width / 2 && Math.abs(localY) <= box.height / 2) {
        return box;
      }
    } else {
      // Standard rectangular check
      if (x >= box.x && x <= box.x + box.width &&
          y >= box.y && y <= box.y + box.height) {
        return box;
      }
    }
  }
  return null;
}

function redrawCanvas() {
  if (originalImageData) {
    ctx.putImageData(originalImageData, 0, 0);
  }
  drawRedactionBoxes();
}

// Mouse events
canvas.addEventListener('mousedown', (e) => {
  if (redactionBoxes.length === 0) return;
  
  const pos = getPointerPos(e);
  draggedBox = findBoxAt(pos.x, pos.y);
  
  if (draggedBox) {
    if (draggedBox.type === 'rotated') {
      // For rotated boxes, offset is from center
      dragOffset.x = pos.x - draggedBox.x;
      dragOffset.y = pos.y - draggedBox.y;
    } else {
      // For standard boxes, offset is from top-left
      dragOffset.x = pos.x - draggedBox.x;
      dragOffset.y = pos.y - draggedBox.y;
    }
    canvas.style.cursor = 'grabbing';
    e.preventDefault();
  }
});

canvas.addEventListener('mousemove', (e) => {
  if (redactionBoxes.length === 0) return;
  
  const pos = getPointerPos(e);
  
  if (draggedBox) {
    draggedBox.x = pos.x - dragOffset.x;
    draggedBox.y = pos.y - dragOffset.y;
    
    // Keep box within canvas bounds
    if (draggedBox.type === 'rotated') {
      // For rotated boxes, constrain center point considering rotation
      const halfDiag = Math.sqrt(draggedBox.width * draggedBox.width + draggedBox.height * draggedBox.height) / 2;
      draggedBox.x = Math.max(halfDiag, Math.min(canvas.width - halfDiag, draggedBox.x));
      draggedBox.y = Math.max(halfDiag, Math.min(canvas.height - halfDiag, draggedBox.y));
    } else {
      // Standard box constraints
      draggedBox.x = Math.max(0, Math.min(canvas.width - draggedBox.width, draggedBox.x));
      draggedBox.y = Math.max(0, Math.min(canvas.height - draggedBox.height, draggedBox.y));
    }
    
    redrawCanvas();
  } else {
    // Change cursor when hovering over a box
    canvas.style.cursor = findBoxAt(pos.x, pos.y) ? 'grab' : 'default';
  }
});

canvas.addEventListener('mouseup', () => {
  draggedBox = null;
  canvas.style.cursor = 'default';
});

canvas.addEventListener('mouseleave', () => {
  draggedBox = null;
  canvas.style.cursor = 'default';
});

// Touch events
canvas.addEventListener('touchstart', (e) => {
  if (redactionBoxes.length === 0) return;
  
  const pos = getPointerPos(e);
  draggedBox = findBoxAt(pos.x, pos.y);
  
  if (draggedBox) {
    if (draggedBox.type === 'rotated') {
      // For rotated boxes, offset is from center
      dragOffset.x = pos.x - draggedBox.x;
      dragOffset.y = pos.y - draggedBox.y;
    } else {
      // For standard boxes, offset is from top-left
      dragOffset.x = pos.x - draggedBox.x;
      dragOffset.y = pos.y - draggedBox.y;
    }
    e.preventDefault();
  }
});

canvas.addEventListener('touchmove', (e) => {
  if (draggedBox && e.touches.length === 1) {
    const pos = getPointerPos(e);
    draggedBox.x = pos.x - dragOffset.x;
    draggedBox.y = pos.y - dragOffset.y;
    
    // Keep box within canvas bounds
    if (draggedBox.type === 'rotated') {
      // For rotated boxes, constrain center point considering rotation
      const halfDiag = Math.sqrt(draggedBox.width * draggedBox.width + draggedBox.height * draggedBox.height) / 2;
      draggedBox.x = Math.max(halfDiag, Math.min(canvas.width - halfDiag, draggedBox.x));
      draggedBox.y = Math.max(halfDiag, Math.min(canvas.height - halfDiag, draggedBox.y));
    } else {
      // Standard box constraints
      draggedBox.x = Math.max(0, Math.min(canvas.width - draggedBox.width, draggedBox.x));
      draggedBox.y = Math.max(0, Math.min(canvas.height - draggedBox.height, draggedBox.y));
    }
    
    redrawCanvas();
    e.preventDefault();
  }
});

canvas.addEventListener('touchend', () => {
  draggedBox = null;
});

canvas.addEventListener('touchcancel', () => {
  draggedBox = null;
});

// Debug mode toggle (press 'd' key)
document.addEventListener('keydown', (e) => {
  if (e.key === 'd' || e.key === 'D') {
    window.debugMode = !window.debugMode;
    console.log('Debug mode:', window.debugMode ? 'ON' : 'OFF');
    if (redactionBoxes.length > 0) {
      redrawCanvas();
    }
  }
});