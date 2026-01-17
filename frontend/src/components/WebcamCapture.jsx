import React, { useState } from "react";
import Webcam from "react-webcam";
import { Camera } from "lucide-react";
import { useWebcam } from "../hooks/useWebcam";

const WebcamCapture = ({ onVerify, isProcessing }) => {
  const { webcamRef, captureRGBFrame } = useWebcam();
  const [status, setStatus] = useState("");

  const startLivenessCheck = async () => {
    setStatus("Capturing liveness...");
    const frames = [];
    const fps = 30;
    const totalFrames = 90;

    // ---- Capture one selfie image (base64) ----
    const selfieImage = webcamRef.current.getScreenshot();

    // ---- Capture RGB frames for rPPG ----
    for (let i = 0; i < totalFrames; i++) {
      const rgb = captureRGBFrame();
      if (rgb) frames.push(rgb);
      await new Promise((res) => setTimeout(res, 1000 / fps));
    }

    setStatus("");

    // ---- Send EVERYTHING to backend ----
    onVerify({
      fps,
      rgb_frames: frames,
      selfie_image: selfieImage,
    });
  };

  return (
    <div className="relative rounded-2xl overflow-hidden border border-white/10 bg-black">
      <Webcam
        audio={false}
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        className="w-full"
      />

      <div className="absolute bottom-6 inset-x-0 flex flex-col items-center gap-2">
        {status && <p className="text-xs text-blue-400">{status}</p>}
        <button
          onClick={startLivenessCheck}
          disabled={isProcessing}
          className="bg-blue-600 px-8 py-3 rounded-full font-bold flex items-center gap-2"
        >
          <Camera size={18} />
          {isProcessing ? "Processing..." : "Verify Living Form"}
        </button>
      </div>
    </div>
  );
};

export default WebcamCapture;
