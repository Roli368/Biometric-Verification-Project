import React from "react";
import Webcam from "react-webcam";
import { Camera } from "lucide-react";

const WebcamCapture = ({ onVerify }) => {
  const webcamRef = React.useRef(null);

  const captureSelfie = async () => {
    const base64 = webcamRef.current.getScreenshot();
    if (!base64) return;

    const blob = await fetch(base64).then((res) => res.blob());
    onVerify({ selfieBlob: blob });
  };

  return (
    <div className="w-full max-w-sm mx-auto space-y-4">
      {/* Webcam container */}
      <div className="relative rounded-xl overflow-hidden border border-white/10 bg-black">
        <Webcam
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          className="w-full h-64 object-cover"
          videoConstraints={{
            facingMode: "user",
          }}
        />
      </div>

      {/* Action button */}
      <button
        onClick={captureSelfie}
        className="w-full bg-blue-600 hover:bg-blue-700 transition px-6 py-3 rounded-full font-semibold flex items-center justify-center gap-2"
      >
        <Camera size={18} />
        Capture & Verify
      </button>
    </div>
  );
};

export default WebcamCapture;

