// 1. YOUR CLASSES
const classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'DEL', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'SPACE', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'];

// 2. SENTENCE BUILDER VARIABLES
let sentence = "";
let lastPred = "";
let framesHeld = 0;
const REQUIRED_FRAMES = 15; // Hold for ~0.5s to type

// DOM Elements
const videoElement = document.getElementById('video');
const canvasElement = document.getElementById('canvas');
const canvasCtx = canvasElement.getContext('2d');

const sentenceOutput = document.getElementById('sentence-output');
const currentGesture = document.getElementById('current-gesture');
const confidenceText = document.getElementById('confidence');
const confidencePill = document.getElementById('confidence-pill');
const typingProgress = document.getElementById('typing-progress');

const startBtn = document.getElementById('start-btn');
const clearBtn = document.getElementById('clear-btn');
const statusBadge = document.getElementById('status-badge');
const cameraPlaceholder = document.getElementById('camera-placeholder');
const trackedBadge = document.getElementById('tracked-badge');

const modelStatusDot = document.getElementById('model-status-dot');
const modelStatusText = document.getElementById('model-status-text');

let session;
let isDetecting = false;

// Load the ONNX model
async function loadModel() {
    try {
        session = await ort.InferenceSession.create('./model_web.onnx');
        console.log("Model loaded!");
        modelStatusDot.classList.replace('bg-amber-400', 'bg-teal-400');
        modelStatusDot.classList.remove('animate-pulse');
        modelStatusText.innerText = 'AI MODEL READY';
    } catch (e) {
        console.error("Failed to load model", e);
        modelStatusDot.classList.replace('bg-amber-400', 'bg-red-500');
        modelStatusText.innerText = 'MODEL ERROR';
        sentenceOutput.innerText = "Error loading AI model.";
    }
}
loadModel();

// Helper: Softmax function to convert AI outputs to percentages
function softmax(arr) {
    const max = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
}

// Process each frame
async function onResults(results) {
    // We only process/draw if the user has clicked start
    if (!isDetecting) return;

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0 && session) {
        
        // Show the tracking badges nicely
        trackedBadge.style.opacity = '1';
        trackedBadge.classList.add('text-teal-600');
        confidencePill.style.opacity = '1';
        confidencePill.style.transform = 'translateY(0)';

        const landmarks = results.multiHandLandmarks[0];
        
        // Draw MediaPipe Hands (Green styling matching the UI)
        drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: '#ffffff', lineWidth: 4 });
        drawLandmarks(canvasCtx, landmarks, { color: '#14b8a6', lineWidth: 2 }); // teal-500

        // Extract coordinates relative to wrist
        let baseX = landmarks[0].x;
        let baseY = landmarks[0].y;
        let baseZ = landmarks[0].z;

        let row = [];
        for (let i = 0; i < landmarks.length; i++) {
            row.push(landmarks[i].x - baseX);
            row.push(landmarks[i].y - baseY);
            row.push(landmarks[i].z - baseZ);
        }

        // Run Inference
        try {
            const tensor = new ort.Tensor('float32', Float32Array.from(row), [1, 63]);
            const output = await session.run({ input: tensor });
            const logits = output.output.data;

            // Calculate probabilities
            const probs = softmax(Array.from(logits));
            let maxIndex = 0;
            let maxProb = probs[0];
            for (let i = 1; i < probs.length; i++) {
                if (probs[i] > maxProb) {
                    maxProb = probs[i];
                    maxIndex = i;
                }
            }

            const predictedLabel = classes[maxIndex];
            const confString = (maxProb * 100).toFixed(1) + "%";

            // Update UI Labels with active colors
            currentGesture.innerText = predictedLabel;
            currentGesture.classList.replace('text-slate-400', 'text-teal-600');
            currentGesture.classList.replace('bg-slate-50', 'bg-teal-50');
            currentGesture.classList.replace('border-slate-100', 'border-teal-200');
            
            confidenceText.innerText = confString;
            confidenceText.classList.replace('text-slate-300', 'text-slate-900');
            
            confidencePill.innerText = "CONFIDENCE " + confString;

            // --- SENTENCE BUILDER LOGIC ---
            if (predictedLabel === lastPred) {
                framesHeld++;
            } else {
                framesHeld = 0;
                lastPred = predictedLabel;
            }

            // Update Progress Bar
            if (framesHeld > 0) {
                let pct = Math.min((framesHeld / REQUIRED_FRAMES) * 100, 100);
                typingProgress.style.width = pct + "%";
            } else {
                typingProgress.style.width = "0%";
            }

            // Type the letter!
            if (framesHeld === REQUIRED_FRAMES) {
                if (predictedLabel === "SPACE") {
                    sentence += " ";
                } else if (predictedLabel === "DEL") {
                    sentence = sentence.slice(0, -1);
                } else {
                    sentence += predictedLabel;
                }
                
                sentenceOutput.innerText = sentence.length > 0 ? `"${sentence}"` : "Waiting for signs...";
                framesHeld = -10; // Cooldown pause before next type
                typingProgress.style.width = "0%";
            }

        } catch (e) {
            console.error("Inference error: ", e);
        }
    } else {
        // Reset if no hand detected
        framesHeld = 0;
        lastPred = "";
        typingProgress.style.width = "0%";
        
        // Dim the UI back to inactive states
        currentGesture.innerText = "-";
        currentGesture.className = "text-sm font-bold text-slate-400 bg-slate-50 px-3 py-1 rounded-full border border-slate-100 transition-colors duration-300";
        confidenceText.innerText = "-";
        confidenceText.className = "font-bold text-slate-300 transition-colors duration-300";
        
        trackedBadge.style.opacity = '0';
        confidencePill.style.opacity = '0';
        confidencePill.style.transform = 'translateY(1rem)';
    }
    canvasCtx.restore();
}

// Setup MediaPipe
const hands = new Hands({
    locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
    }
});
hands.setOptions({
    selfieMode: true, // Mirrors camera to match training data
    maxNumHands: 1,
    modelComplexity: 0,
    minDetectionConfidence: 0.7
});
hands.onResults(onResults);

const camera = new Camera(videoElement, {
    onFrame: async () => { 
        if(isDetecting) {
            await hands.send({ image: videoElement }); 
        }
    },
    width: 640, height: 853 // 3:4 aspect ratio
});

// --- BUTTON LISTENERS ---

// Toggle Camera
startBtn.addEventListener('click', () => {
    if (!isDetecting) {
        // TURN CAMERA ON
        camera.start();
        isDetecting = true;
        
        // Hide Placeholder
        cameraPlaceholder.style.opacity = '0';
        setTimeout(() => cameraPlaceholder.style.visibility = 'hidden', 300);

        // Update Button to "Stop"
        startBtn.innerText = "Stop Detection";
        startBtn.className = "w-full bg-gradient-to-r from-rose-500 to-red-500 text-white font-bold text-lg py-4 rounded-[1.25rem] shadow-md hover:from-rose-600 hover:to-red-600 hover:shadow-lg transition-all duration-200";

        // Update Status Badge
        statusBadge.innerText = "● Live";
        statusBadge.className = "text-xs font-semibold text-teal-600 bg-teal-50 border border-teal-100 px-2.5 py-1 rounded-md transition-colors duration-300";
        
        if(sentence.length === 0) sentenceOutput.innerText = "Waiting for signs...";
        
    } else {
        // TURN CAMERA OFF
        isDetecting = false;
        camera.stop();

        // Turn off physical webcam light
        if (videoElement.srcObject) {
            videoElement.srcObject.getTracks().forEach(track => track.stop());
            videoElement.srcObject = null;
        }

        // Clear Canvas & Show Placeholder
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        cameraPlaceholder.style.visibility = 'visible';
        cameraPlaceholder.style.opacity = '1';
        
        // Reset Button to "Start"
        startBtn.innerText = "Start Detection";
        startBtn.className = "w-full bg-gradient-to-r from-indigo-500 to-teal-500 text-white font-bold text-lg py-4 rounded-[1.25rem] shadow-md hover:from-indigo-600 hover:to-teal-600 hover:shadow-lg transition-all duration-200";

        // Reset Status Badge
        statusBadge.innerText = "● Standby";
        statusBadge.className = "text-xs font-semibold text-slate-500 bg-slate-100 px-2.5 py-1 rounded-md transition-colors duration-300";
        
        // Hide Tracking Badges
        trackedBadge.style.opacity = '0';
        confidencePill.style.opacity = '0';
        currentGesture.innerText = "-";
        currentGesture.className = "text-sm font-bold text-slate-400 bg-slate-50 px-3 py-1 rounded-full border border-slate-100 transition-colors duration-300";
        confidenceText.innerText = "-";
        confidenceText.className = "font-bold text-slate-300 transition-colors duration-300";
    }
});

// Clear Text Button
clearBtn.addEventListener('click', () => {
    sentence = "";
    sentenceOutput.innerText = isDetecting ? "Waiting for signs..." : "Camera is stopped.";
});

// Keyboard shortcut: Press 'C' to clear sentence
window.addEventListener('keydown', (e) => {
    if (e.key.toLowerCase() === 'c') {
        clearBtn.click();
    }
});
